# flow_stats_collector.py

import csv
import os
import time
import logging

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub

LOG = logging.getLogger('flow_stats_collector')

POLL_INTERVAL = 5
ATTACKER_IP   = '10.0.0.4'          # h4 is the attacker 
MONITORED_SRC_HOSTS = ['10.0.0.1', '10.0.0.2', '10.0.0.3', '10.0.0.4']
OUTPUT_CSV    = 'flows.csv'

PRIORITY_IPFLOW = 10

# ip_proto values we install explicit monitoring flows for
PROTO_NUMS = {'TCP': 6, 'UDP': 17, 'ICMP': 1}

CSV_FIELDS = [
    'timestamp', 'src_ip', 'proto',
    'pps', 'bps', 'duration', 'packet_count', 'byte_count', 'label'
]


def _init_csv(path):
    if not os.path.exists(path):
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(CSV_FIELDS)
        LOG.info('Created %s', path)
    else:
        LOG.info('Appending to existing %s', path)


def _append_rows(path, rows):
    if not rows:
        return
    with open(path, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerows(rows)


class FlowStatsCollector(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datapaths = {}
        self.prev_stats = {}
        self.monitor_thread = hub.spawn(self._monitor_loop)
        _init_csv(OUTPUT_CSV)
        LOG.info('Collector started — polling every %ds → %s',
                 POLL_INTERVAL, OUTPUT_CSV)

    # ── Polling loop 

    def _monitor_loop(self):
        while True:
            hub.sleep(POLL_INTERVAL)
            for dp in list(self.datapaths.values()):
                self._request_stats(dp)

    def _request_stats(self, datapath):
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)
        LOG.debug('Stats request → dpid=%016x', datapath.id)

    # ── Switch handshake 

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto  = datapath.ofproto
        parser   = datapath.ofproto_parser
        dpid     = datapath.id

        self.datapaths[dpid] = datapath

        # Table-miss → controller 
        self._add_flow(datapath, 0, parser.OFPMatch(),
                       [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                               ofproto.OFPCML_NO_BUFFER)])

        for src in MONITORED_SRC_HOSTS:
            for proto_num in PROTO_NUMS.values():
                match = parser.OFPMatch(
                    eth_type=0x0800, ipv4_src=src, ip_proto=proto_num
                )
                actions = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]
                self._add_flow(datapath, PRIORITY_IPFLOW, match, actions)

        LOG.info('Switch connected dpid=%016x — per-protocol IP flows pre-installed', dpid)

    # ── PacketIn 

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

    # ── Stats reply 

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        body      = ev.msg.body
        now       = time.time()
        rows      = []
        attack_n  = 0
        normal_n  = 0
        seen_keys = set()

        for stat in body:
            if stat.priority != PRIORITY_IPFLOW:
                continue

            match     = stat.match
            src_ip    = match.get('ipv4_src', None)
            proto_num = match.get('ip_proto', None)
            if src_ip is None or proto_num is None:
                continue

            proto_str = {6: 'TCP', 17: 'UDP', 1: 'ICMP'}.get(proto_num, str(proto_num))
            key = (src_ip, proto_str)
            seen_keys.add(key)

            duration = stat.duration_sec + stat.duration_nsec / 1e9
            packet_count = stat.packet_count
            byte_count   = stat.byte_count

            prev = self.prev_stats.get(key)
            self.prev_stats[key] = (packet_count, byte_count, now)

            if prev is None:
                continue

            prev_packets, prev_bytes, prev_time = prev
            dt = now - prev_time
            if dt <= 0:
                continue

            dpackets = packet_count - prev_packets
            dbytes   = byte_count - prev_bytes

            if dpackets <= 0:
                continue

            pps = round(dpackets / dt, 4)
            bps = round(dbytes / dt, 4)
            label = 1 if src_ip == ATTACKER_IP else 0

            if label == 1:
                attack_n += 1
            else:
                normal_n += 1

            rows.append({
                'timestamp':    round(now, 3),
                'src_ip':       src_ip,
                'proto':        proto_str,
                'pps':          pps,
                'bps':          bps,
                'duration':     round(duration, 4),
                'packet_count': dpackets,   
                'byte_count':   dbytes,     
                'label':        label,
            })

        # Clean up stale keys
        stale = set(self.prev_stats.keys()) - seen_keys
        for key in stale:
            del self.prev_stats[key]

        if rows:
            _append_rows(OUTPUT_CSV, rows)
            LOG.info('Wrote %d rows → %s  (attack=%d  normal=%d)',
                     len(rows), OUTPUT_CSV, attack_n, normal_n)
        else:
            LOG.info('No delta rows this cycle — waiting for a second poll per flow...')
