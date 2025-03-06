#!/bin/bash
ARG_A=${ARG_A}
CMD=$(python3 utils/mahi_helpers.py)
if [ -z "$CMD" ]; then
  echo "Error: CMD is empty!"
  exit 1
fi

echo "Generated mahi command: $CMD"
echo "user: $whoami"
echo "Algorithm: $ARG_A"
# exec $CMD -- python run.py --sender -A "$ARG_A"

#  Enable IP forwarding in the host container
sudo sysctl -w net.ipv4.ip_forward=1 > /app/share/ip_forward.log 2>&1

# Create networking setup commands that use MAHIMAHI_BASE
NETWORK_SETUP="echo \"MAHIMAHI_BASE=\$MAHIMAHI_BASE\" > /app/share/mahimahi_vars.log && \
sudo ip route add 192.168.2.0/24 via \$MAHIMAHI_BASE && \
sudo iptables -A FORWARD -d 192.168.2.0/24 -j ACCEPT && \
sudo iptables -A FORWARD -s 192.168.2.0/24 -j ACCEPT && \
sudo iptables -t nat -A POSTROUTING -d 192.168.2.0/24 -j MASQUERADE && \
ip route show > /app/share/mahimahi_routes.log && \
ping -c 2 192.168.2.102 > /app/share/ping_test.log 2>&1 || true"

# Complete mahimahi command
FINAL_CMD="$CMD -- sh -c \"$NETWORK_SETUP && python run.py --sender -A \\\"$ARG_A\\\"\""

echo "Executing: $FINAL_CMD"

# Execute the final command
eval "$FINAL_CMD"