source modules/containernet/venv/bin/activate
sudo -E env PATH=$PATH python topo/topo_one2one.py -T trace_10k.json -A dummy
sudo -E env PATH=$PATH python topo/topo_one2one.py -T trace_10k.json -A GCC
#sudo -E env PATH=$PATH python topo/topo_one2one.py -T trace_10k.json -A HRCC
deactivate