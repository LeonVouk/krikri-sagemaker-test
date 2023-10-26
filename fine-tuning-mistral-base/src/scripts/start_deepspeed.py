"""
start training job
"""
import os
import json
import socket
from huggingface_hub import HfFolder


if __name__ == "__main__":

    if HfFolder.get_token() is not None:
        # huggingface token to access gated models, e.g. llama 2
        os.environ['HF_TOKEN'] = HfFolder.get_token()

    # hosts = json.loads(os.environ['SM_HOSTS'])
    # current_host = os.environ['SM_CURRENT_HOST']
    # host_rank = int(hosts.index(current_host))

    # master = json.loads(os.environ['SM_TRAINING_ENV'])['master_hostname']
    # master_addr = socket.gethostbyname(master)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    # os.environ['NODE_INDEX'] = str(host_rank)
    # os.environ['SM_MASTER'] = str(master)
    # os.environ['SM_MASTER_ADDR'] = str(master_addr)
    # os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'

    os.system("chmod +x ./torch_launch.sh")
    os.system("/bin/bash -c ./torch_launch.sh")
