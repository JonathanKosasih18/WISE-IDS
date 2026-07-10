#!/usr/bin/env python3
# topology.py 

from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import setLogLevel


def build_topology():
    net = Mininet(controller=RemoteController, switch=OVSKernelSwitch, link=TCLink)

    print("*** Adding controller")
    c0 = net.addController('c0', controller=RemoteController, ip='127.0.0.1', port=6633)

    print("*** Adding switch")
    s1 = net.addSwitch('s1', protocols='OpenFlow13')

    print("*** Adding hosts")
    h1 = net.addHost('h1', ip='10.0.0.1/24')  # normal
    h2 = net.addHost('h2', ip='10.0.0.2/24')  # normal
    h3 = net.addHost('h3', ip='10.0.0.3/24')  # normal
    h4 = net.addHost('h4', ip='10.0.0.4/24')  # attacker
    h5 = net.addHost('h5', ip='10.0.0.5/24')  # server / victim

    print("*** Creating shared links (all hosts on same switch = shared bandwidth)")
    for h in [h1, h2, h3, h4, h5]:
        net.addLink(h, s1, bw=100)  

    print("*** Starting network")
    net.build()
    c0.start()
    s1.start([c0])

    return net


if __name__ == '__main__':
    setLogLevel('info')
    net = build_topology()
    CLI(net)
    net.stop()
