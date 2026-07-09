# ids_ryu_app.py
# Day 3 Lab — AI/ML IDS on SDN
# WISE Lab Internship Prep
#
# What this does:
#   - Loads ids.joblib (RandomForestClassifier) + proto_encoder.joblib at startup
#   - Acts as an L2 switch (floods on PacketIn, same as flow_stats_collector.py)
#   - Polls OFPFlowStatsRequest every 5s
#   - For each flow: extracts [pps, bps, duration, proto_enc] features
#   - Runs model.predict() — if attack (1), installs a DROP flow for that
#     src_ip with hard_timeout=60s
#   - Logs every detection and every auto-block action
#
# Run with:
#   ryu-manager ids_ryu_app.py
#
# Requires (same directory as this script):
#   ids.joblib            — trained RandomForestClassifier
#   proto_encoder.joblib  — LabelEncoder fit on the 'proto' column
#   feature_cols.txt      — feature column order used during training
#
# Topology assumption (Day 4, integrated normal+attack testbed):
#   h1=10.0.0.1 normal (iperf3 client)   h2=10.0.0.2 normal (iperf3 client)
#   h3=10.0.0.3 normal (iperf3 client)   h4=10.0.0.4 attacker (hping3)
#   h5=10.0.0.5 shared server/victim (iperf3 server + hping3/ping target)
#
# Note: h5 is a destination only (never a traffic source), so it does not need
# a pre-installed ipv4_src flow — the four sender hosts (h1-h4) below are
# unchanged in address from the Week 1 assumption, just relabeled by role.

import os
import time
import logging

import joblib
import numpy as np
import pandas as pd

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub

LOG = logging.getLogger('ids_ryu_app')

# ── Configuration ─────────────────────────────────────────────────────────────
POLL_INTERVAL   = 5          # seconds between OFPFlowStatsRequest polls
BLOCK_TIMEOUT   = 60         # seconds the DROP rule stays installed
PRIORITY_BLOCK  = 100        # must outrank the catch-all IP flows (priority 10)
PRIORITY_IPFLOW = 10
PRIORITY_MISS   = 0

MODEL_PATH        = 'ids_v2.joblib'
ENCODER_PATH      = 'proto_encoder_v2.joblib'
FEATURE_COLS_PATH = 'feature_cols_v2.txt'

PROTO_MAP = {1: 'ICMP', 6: 'TCP', 17: 'UDP'}


def _load_feature_cols(path):
    if not os.path.exists(path):
        # Fallback to the known training order if file is missing
        LOG.warning('%s not found — using default feature order', path)
        return ['pps', 'bps', 'duration', 'proto_enc']
    with open(path) as f:
        cols = [line.strip() for line in f if line.strip()]
    return cols


class IDSRyuApp(app_manager.RyuApp):
    """
    Ryu controller with embedded ML-based IDS.

    Pipeline each poll cycle:
      stats collection -> feature extraction -> normalization (none needed,
      RF is scale-invariant) -> inference -> auto-block -> logging
    """

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datapaths   = {}
        self.blocked_ips = {}   # {src_ip: block_timestamp} — avoid reblocking spam

        # Load model artifacts
        LOG.info('Loading IDS model artifacts...')
        self.model        = joblib.load(MODEL_PATH)
        self.proto_encoder = joblib.load(ENCODER_PATH)
        self.feature_cols = _load_feature_cols(FEATURE_COLS_PATH)
        LOG.info('Model loaded. Features (in order): %s', self.feature_cols)
        LOG.info('Known protocols: %s', list(self.proto_encoder.classes_))

        self.monitor_thread = hub.spawn(self._monitor_loop)
        LOG.info('IDS started — polling every %ds, block_timeout=%ds',
                 POLL_INTERVAL, BLOCK_TIMEOUT)

    # ── Background polling loop ───────────────────────────────────────────────

    def _monitor_loop(self):
        while True:
            hub.sleep(POLL_INTERVAL)
            for dp in list(self.datapaths.values()):
                self._request_stats(dp)

    def _request_stats(self, datapath):
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    # ── Switch handshake ──────────────────────────────────────────────────────

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto  = datapath.ofproto
        parser   = datapath.ofproto_parser
        dpid     = datapath.id

        self.datapaths[dpid] = datapath

        # Table-miss → controller
        self._add_flow(datapath, PRIORITY_MISS, parser.OFPMatch(),
                       [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                               ofproto.OFPCML_NO_BUFFER)])

        # Pre-install per-host, PER-PROTOCOL IP flows so stats always include
        # both ipv4_src AND ip_proto (needed for inference).
        #
        # IMPORTANT: a flow's `match` in an OFPFlowStatsReply only contains
        # the fields that were part of THAT flow's original match criteria.
        # The previous version only matched on (eth_type, ipv4_src) -- ip_proto
        # was never part of the match, so stat.match.get('ip_proto', 0) always
        # returned the default 0, which PROTO_MAP mapped to 'any' regardless
        # of whether the real traffic was TCP/UDP/ICMP. This silently broke
        # proto-aware classification for ALL protocols in production, not
        # just UDP -- confirmed via live testing (2026-07-09) where even a
        # SYN-flood attack (previously assumed to be scored with proto=TCP)
        # showed proto=any in the logs before this fix.
        #
        # Fix: install THREE flows per host (one per protocol we care about)
        # instead of one flow matching all IP traffic. Each flow's own match
        # now legitimately contains ip_proto, so OVS reports it back
        # correctly per-protocol, per-host in flow stats.
        #
        # h1-h3 = normal traffic sources, h4 = attacker. h5 (server) is a
        # destination only and intentionally excluded from this src-IP list.
        hosts = ['10.0.0.1', '10.0.0.2', '10.0.0.3', '10.0.0.4']
        monitored_protocols = [1, 6, 17]   # ICMP, TCP, UDP — keys of PROTO_MAP
        for src in hosts:
            for proto_num in monitored_protocols:
                match = parser.OFPMatch(eth_type=0x0800, ipv4_src=src,
                                        ip_proto=proto_num)
                actions = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]
                self._add_flow(datapath, PRIORITY_IPFLOW, match, actions)

        LOG.info('Switch connected dpid=%016x — IDS active (%d hosts x %d '
                 'protocols monitored)', dpid, len(hosts), len(monitored_protocols))

    # ── PacketIn — basic flood forwarding (ARP etc.) ──────────────────────────

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg      = ev.msg
        datapath = msg.datapath
        ofproto  = datapath.ofproto
        parser   = datapath.ofproto_parser
        in_port  = msg.match['in_port']

        actions = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=ofproto.OFP_NO_BUFFER,
            in_port=in_port,
            actions=actions,
            data=msg.data,
        )
        datapath.send_msg(out)

    def _add_flow(self, datapath, priority, match, actions,
                  idle_timeout=0, hard_timeout=0):
        ofproto = datapath.ofproto
        parser  = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(
            ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=priority,
            match=match,
            instructions=inst,
            idle_timeout=idle_timeout,
            hard_timeout=hard_timeout,
        )
        datapath.send_msg(mod)

    # ── Auto-block: install DROP rule for an offending src IP ────────────────

    def _block_ip(self, datapath, src_ip):
        """Install a DROP flow for src_ip with hard_timeout=60s."""
        now = time.time()

        # Avoid spamming duplicate block installs every poll cycle
        last_blocked = self.blocked_ips.get(src_ip)
        if last_blocked and (now - last_blocked) < BLOCK_TIMEOUT:
            LOG.debug('%s already blocked (%.0fs ago) — skipping reinstall',
                      src_ip, now - last_blocked)
            return

        parser = datapath.ofproto_parser
        match = parser.OFPMatch(eth_type=0x0800, ipv4_src=src_ip)
        actions = []   # empty actions list = DROP

        self._add_flow(
            datapath, PRIORITY_BLOCK, match, actions,
            hard_timeout=BLOCK_TIMEOUT
        )
        self.blocked_ips[src_ip] = now

        LOG.warning(
            '🚫 AUTO-BLOCK: src_ip=%s flagged as ATTACK — DROP rule installed '
            '(priority=%d, hard_timeout=%ds)',
            src_ip, PRIORITY_BLOCK, BLOCK_TIMEOUT
        )

    # ── Flow stats reply → feature extraction → inference ─────────────────────

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        body     = ev.msg.body
        datapath = ev.msg.datapath
        checked  = 0
        flagged  = 0
        cycle_status = {}   # {src_ip: 'ATTACK'/'normal'} — for the per-cycle summary log

        for stat in body:
            if stat.priority != PRIORITY_IPFLOW:
                continue   # only inspect our pre-installed IP flows

            match  = stat.match
            src_ip = match.get('ipv4_src', None)
            if src_ip is None:
                continue

            proto_num = match.get('ip_proto', 0)
            proto_str = PROTO_MAP.get(proto_num, 'any') if proto_num else 'any'

            duration = stat.duration_sec + stat.duration_nsec / 1e9
            if duration <= 0:
                duration = 1.0

            pps = stat.packet_count / duration
            bps = stat.byte_count   / duration

            # Skip flows with no real traffic yet
            if stat.packet_count == 0:
                continue

            # Encode protocol — handle unseen labels gracefully
            try:
                proto_enc = int(self.proto_encoder.transform([proto_str])[0])
            except ValueError:
                LOG.warning('⚠️  Unseen proto "%s" for %s — proto_encoder only '
                           'knows %s. Defaulting proto_enc=0 (collides with '
                           'first known class). Retrain proto_encoder if this '
                           'protocol should be recognized.',
                           proto_str, src_ip, list(self.proto_encoder.classes_))
                proto_enc = 0

            feature_row = {
                'pps': pps,
                'bps': bps,
                'duration': duration,
                'proto_enc': proto_enc,
            }
            X = pd.DataFrame([feature_row])[self.feature_cols]

            prediction = self.model.predict(X)[0]
            confidence = max(self.model.predict_proba(X)[0])
            checked += 1
            cycle_status[src_ip] = 'ATTACK' if prediction == 1 else 'normal'

            LOG.debug(
                'Inference: src=%s pps=%.1f bps=%.1f proto=%s -> %s (%.2f)',
                src_ip, pps, bps, proto_str,
                'ATTACK' if prediction == 1 else 'normal', confidence
            )

            if prediction == 1:
                flagged += 1
                LOG.warning(
                    '⚠️  DETECTED: src=%s pps=%.1f bps=%.1f proto=%s '
                    'confidence=%.2f',
                    src_ip, pps, bps, proto_str, confidence
                )
                self._block_ip(datapath, src_ip)

        if checked:
            status_str = ', '.join(f'{ip}={status}' for ip, status in sorted(cycle_status.items()))
            LOG.info('Poll cycle: checked=%d flagged=%d blocked_total=%d | %s',
                     checked, flagged, len(self.blocked_ips), status_str)
