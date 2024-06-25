# *BiomedParse* 
:grapes: \[[Read Our arXiv Paper](https://arxiv.org/abs/2405.12971)\] &nbsp; :apple: \[[Check Our Demo](https://microsoft.github.io/BiomedParse/)\] 

install docker
```
sudo apt update
sudo apt install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt update
apt-cache policy docker-ce
sudo apt install docker-ce
```

Prepare docker environment:
Specify project directories in docker/README.md
```sh
bash docker/docker_build.sh
bash docker/docker_run.sh
bash docker/setup_inside_docker.sh
source docker/data_env.sh 
```

Training using example BioParseData:
```sh
bash assets/scripts/train.sh
```

Evaluation on example BioParseData:
```sh
bash assets/scripts/eval.sh
```

Detection and Recognition inference code are provided in inference_utils/output_processing.py. check_mask_stats() outputs p value for model predicted mask for detection. combine_masks() combines predictions for non-overlapping masks. See the Method section in our paper for details.
