#!/usr/bin/env python3
"""
day6_capture_normal.py — Day 6 (tracker item 2c): capture throughput/loss/delay
for NORMAL traffic (h1/h2/h3 -> h5) using concurrent iperf3 UDP + ping.

Run this INSTEAD of `topology.py` for this capture (it builds the same topology
internally via build_topology(), then drives the traffic automatically instead
of dropping into the CLI).

Usage:
    sudo python3 day6_capture_normal.py
    (Run your Ryu controller separately first if you want it active during capture:
     ryu-manager ids_ryu_app.py)

Output (written to ./day6/normal_logs/):
    h1_iperf.json, h2_iperf.json, h3_iperf.json  — iperf3 --json UDP output
    h1_ping.txt,   h2_ping.txt,   h3_ping.txt     — raw ping output

Design notes:
- UDP mode (`-u`) so iperf3 reports lost_percent directly per interval (TCP does not).
- Each client gets its OWN iperf3 server port (5201/5202/5203) on h5, with a
  dedicated `iperf3 -s` instance per port. Three concurrent UDP clients sharing
  a single port collide on the control channel -- sessions degrade or end early
  unpredictably (confirmed empirically: h2/h3 cut off ~21s into a 30s capture
  while h1 alone showed erratic throughput/RTT, all on port 5201 default).
- BANDWIDTH is a *target* rate per client, not a cap on the link — with 3 clients
  at 10M each you get ~30 Mbps combined against a 100 Mbps shared switch, which is
  real but modest contention. Raise this later if you want to see loss climb.
- ping runs at the same 1s interval as iperf3 so both can be plotted on a shared
  time axis without resampling.
"""

import os
import time
from mininet.log import setLogLevel, info
from topology import build_topology

OUT_DIR = 'day6/normal_logs'
DURATION = 600      # seconds of traffic
BANDWIDTH = '10M'  # target UDP rate per client -- adjust if you want more/less contention
INTERVAL = 1        # seconds per report interval (iperf3 -i and ping -i)
SERVER_IP = '10.0.0.5'  # h5
BASE_PORT = 5201       # h1->5201, h2->5202, h3->5203


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    net = build_topology()

    h1, h2, h3, h5 = net.get('h1', 'h2', 'h3', 'h5')
    clients = [h1, h2, h3]
    ports = [BASE_PORT + i for i in range(len(clients))]  # 5201, 5202, 5203

    info('*** Starting one iperf3 UDP server per port on h5\n')
    for port in ports:
        h5.cmd(f'iperf3 -s -p {port} -D')  # daemonized, one process per port
    time.sleep(1)

    info('*** Starting concurrent UDP iperf3 clients (h1,h2,h3) + ping, one port each\n')
    for i, (h, port) in enumerate(zip(clients, ports), start=1):
        json_path = os.path.join(OUT_DIR, f'h{i}_iperf.json')
        ping_path = os.path.join(OUT_DIR, f'h{i}_ping.txt')

        # UDP client: -u, target bandwidth -b, duration -t, interval -i, dedicated
        # server port -p, JSON output to file
        h.cmd(f'iperf3 -c {SERVER_IP} -p {port} -u -b {BANDWIDTH} -t {DURATION} -i {INTERVAL} '
              f'--json > {json_path} 2>&1 &')

        # Concurrent ping for delay, 1 sample/sec to match iperf3 interval.
        # -D prints a real Unix timestamp per line -- parse_ping() uses this
        # rather than icmp_seq, since icmp_seq only approximates elapsed time
        # when RTT stays well under 1s (true here, but not for Day 7's attack
        # captures where RTT can run into the seconds).
        h.cmd(f'ping -D -i {INTERVAL} -c {DURATION} {SERVER_IP} > {ping_path} 2>&1 &')

    info(f'*** Traffic running for {DURATION}s...\n')
    time.sleep(DURATION + 3)  # buffer so background procs finish writing output

    info(f'*** Capture complete. Logs in ./{OUT_DIR}/\n')
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    run()
