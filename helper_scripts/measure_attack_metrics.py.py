#!/usr/bin/env python3
"""
day7_capture_attack.py — Day 7 (tracker item 2d/2f): capture throughput/loss/
delay for ATTACK traffic (h4 hping3 SYN flood -> h5), instrumented via live OVS
flow-stat polling (since hping3 --flood suppresses per-packet output) plus a
concurrent ping.

IMPORTANT: run your Ryu controller first, same as Day 6:
    ryu-manager ids_ryu_app.py
The switch needs a controller connection to forward anything, and this capture
also exercises the real IDS detection/block path (tracker item 2e) -- it is not
just raw traffic generation.

BEFORE/AFTER MITIGATION COMPARISON (item 2f):
This script itself doesn't control blocking -- that's the ENABLE_BLOCKING flag
in ids_ryu_app.py. Run this script TWICE, once for each side of the comparison,
making sure the controller's flag matches:

    "before" run (item 2d baseline, unmitigated):
        1. In ids_ryu_app.py, set ENABLE_BLOCKING = False
        2. Restart the controller: ryu-manager ids_ryu_app.py
        3. Run:  sudo python3 day7_capture_attack.py before
        -> writes to ./day7/attack_logs_before/
        -> detection still logs (⚠️ DETECTED lines) but no DROP rule installs,
           so throughput should stay high and loss should stay ~0% the whole
           100s (aside from normal congestion, if any).

    "after" run (item 2f comparison, mitigated -- this supersedes the original
    Day 7 capture, since that one already had blocking active the whole time):
        1. In ids_ryu_app.py, set ENABLE_BLOCKING = True
        2. Restart the controller: ryu-manager ids_ryu_app.py
        3. Run:  sudo python3 day7_capture_attack.py after
        -> writes to ./day7/attack_logs_after/
        -> expect the detect -> block -> auto-recover pattern (throughput/loss
           drop to 0%/100% around t=20s, recover around t=78s per the prior run)

The MODE argument only controls the output directory name here -- it does NOT
change controller behavior. Double check the controller's own log line at
startup ("ENABLE_BLOCKING=True/False") to confirm the flag actually matches
before you start a capture; a mismatch will silently give you two "after"
runs or two "before" runs instead of a real comparison pair.

Usage:
    sudo python3 day7_capture_attack.py [before|after]
    (defaults to "after" if omitted, for backwards compatibility with the
    original Day 7 invocation)

Output (./day7/attack_logs_<mode>/):
    h4_flowstats.csv  — per-second: time_s, allowed_pkts_delta, allowed_bytes_delta,
                         dropped_pkts_delta, dropped_bytes_delta (from live
                         `ovs-ofctl dump-flows` polling of s1, filtered to
                         ipv4_src=10.0.0.4)
    h4_ping.txt       — raw ping output (h4 -> h5), same format as Day 6

Design notes:
- "Throughput" here means DELIVERED throughput: derived from the delta of the
  priority-10 (allowed/monitored) flow entry's packet/byte counters for
  ipv4_src=10.0.0.4. Once the IDS installs its priority-100 DROP rule (in the
  "after" / mitigated run), that entry's counters freeze and delivered
  throughput correctly reads ~0. In the "before" / unmitigated run, no DROP
  rule is ever installed, so this should track the full flood rate for the
  entire capture. This reuses the delta-based approach validated in Week 1
  (Day 4/5), but reads it independently from raw OVS state rather than
  trusting the IDS app's own internal bookkeeping.
- "Loss" here means the fraction of h4's packets that hit the DROP rule vs.
  total seen at the switch that interval -- i.e. this IS a direct measurement
  of the blocking mechanism (item 2e) in action, not ordinary network-congestion
  loss. In the "after" run, expect 0% before detection, jumping toward 100%
  once the block installs. In the "before" run, expect ~0% throughout (no DROP
  rule ever installs to measure).
- Delay comes from a concurrent `ping -D` (h4 -> h5). The `-D` flag prints a
  Unix timestamp on each reply line, which we use for the real elapsed time --
  NOT icmp_seq. Under flood congestion RTTs can run into the thousands of ms,
  so a reply can arrive many seconds after its request was sent; using icmp_seq
  as a time proxy (fine at Day 6's sub-10ms RTTs) badly distorts the timeline
  once RTT exceeds ~1s, which is exactly what happens here.
  In the "after" run, once blocked, ping (ICMP) is dropped by the same rule
  (the block matches all protocols for the flagged src_ip), so expect RTT
  samples to simply stop appearing for that stretch -- a gap in the plot, not
  a zero value. In the "before" run, ping should continue throughout, though
  RTT may still degrade under sustained flood congestion.
- DURATION defaults to 100s (not 30s) specifically so a single "after" capture
  spans the full detect -> block -> auto-recover cycle: detection typically
  fires within the IDS's own 5s poll interval, the block then holds for
  hard_timeout=60s (see `_block_ip` in ids_ryu_app.py), so ~100s leaves a
  comfortable buffer to see traffic resume after the block expires. The
  "before" run doesn't need this margin functionally, but uses the same
  DURATION so the two plots are directly comparable on the same time axis.
- h1/h2/h3 normal traffic is NOT run concurrently here, to keep the attack
  signal isolated and easy to read (combined normal+attack coexistence was
  already validated in Week 1, Day 4/5).
"""

import os
import re
import sys
import time
from mininet.log import setLogLevel, info
from topology import build_topology

DURATION = 600         # seconds of attack traffic -- long enough to see block + auto-recovery
POLL_INTERVAL = 1      # seconds between ovs-ofctl polls (finer-grained than the IDS's own 5s)
ATTACKER_IP = '10.0.0.4'
SERVER_IP = '10.0.0.5'


def poll_flow_counts(switch):
    """Sum n_packets/n_bytes for ATTACKER_IP flows on the switch, split into
    allowed (priority=10, monitored) vs. blocked (priority=100, DROP) entries."""
    out = switch.cmd('ovs-ofctl -O OpenFlow13 dump-flows s1')
    allowed_pkts = allowed_bytes = dropped_pkts = dropped_bytes = 0

    for line in out.splitlines():
        if f'nw_src={ATTACKER_IP}' not in line:
            continue
        pkt_m = re.search(r'n_packets=(\d+)', line)
        byte_m = re.search(r'n_bytes=(\d+)', line)
        prio_m = re.search(r'priority=(\d+)', line)
        if not (pkt_m and byte_m and prio_m):
            continue
        pkts, byts, prio = int(pkt_m.group(1)), int(byte_m.group(1)), int(prio_m.group(1))
        if prio >= 100:
            dropped_pkts += pkts
            dropped_bytes += byts
        else:
            allowed_pkts += pkts
            allowed_bytes += byts

    return allowed_pkts, allowed_bytes, dropped_pkts, dropped_bytes


def run(mode):
    out_dir = f'day7/attack_logs_{mode}'
    os.makedirs(out_dir, exist_ok=True)
    net = build_topology()

    h4, s1 = net.get('h4', 's1')

    info(f'*** MODE={mode} — make sure ids_ryu_app.py has ENABLE_BLOCKING '
         f'{"= True" if mode == "after" else "= False"} and the controller '
         f'was restarted after the last edit\n')

    info('*** Starting hping3 SYN flood (h4 -> h5)\n')
    h4.cmd(f'hping3 -S -p 80 --flood {SERVER_IP} > /dev/null 2>&1 &')

    ping_path = os.path.join(out_dir, 'h4_ping.txt')
    info('*** Starting concurrent ping (h4 -> h5) for delay measurement\n')
    h4.cmd(f'ping -D -i {POLL_INTERVAL} -c {DURATION} {SERVER_IP} > {ping_path} 2>&1 &')

    info('*** Polling OVS flow stats every %ds for %ds\n' % (POLL_INTERVAL, DURATION))
    csv_path = os.path.join(out_dir, 'h4_flowstats.csv')
    prev_allowed_pkts = prev_allowed_bytes = 0
    prev_dropped_pkts = prev_dropped_bytes = 0

    with open(csv_path, 'w') as f:
        f.write('time_s,allowed_pkts_delta,allowed_bytes_delta,dropped_pkts_delta,dropped_bytes_delta\n')
        for t in range(1, DURATION + 1):
            time.sleep(POLL_INTERVAL)
            a_pkts, a_bytes, d_pkts, d_bytes = poll_flow_counts(s1)

            da_pkts = max(0, a_pkts - prev_allowed_pkts)
            da_bytes = max(0, a_bytes - prev_allowed_bytes)
            dd_pkts = max(0, d_pkts - prev_dropped_pkts)
            dd_bytes = max(0, d_bytes - prev_dropped_bytes)

            f.write(f'{t},{da_pkts},{da_bytes},{dd_pkts},{dd_bytes}\n')

            prev_allowed_pkts, prev_allowed_bytes = a_pkts, a_bytes
            prev_dropped_pkts, prev_dropped_bytes = d_pkts, d_bytes

    info('*** Killing hping3 flood on h4\n')
    h4.cmd('pkill hping3')
    time.sleep(2)

    info(f'*** Capture complete. Logs in ./{out_dir}/\n')
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    mode = sys.argv[1] if len(sys.argv) > 1 else 'after'
    if mode not in ('before', 'after'):
        print(f'Usage: sudo python3 day7_capture_attack.py [before|after]')
        sys.exit(1)
    run(mode)
