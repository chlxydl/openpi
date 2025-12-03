## 1. az login
```bash
az login # login with you sc-account
```
## 2. start docker container
```bash
./dk.sh start #|stop|restart # start/stop/restart docker container
```
## 3. list usable cluster
```bash
amlt tl sing # list available cluster
```
you might see
```
TARGET_NAME        RESOURCE_GROUP          SUBSCRIPTION             DEFAULT_WORKSPACE    ACCELERATORS
-----------------  ----------------------  -----------------------  -------------------  ----------------------
msroctobasicvc     gcr-singularity-octo    Singularity Shared OCTO  .                    A100, MI300X
msroctovc          gcr-singularity-octo    Singularity Shared OCTO  .                    A100, MI300X
msrresrchbasicvc   gcr-singularity         Singularity Shared       .                    A100, CPU, H100, MI200
msrresrchlab       gcr-singularity-lab     Singularity Lab          .                    V100
msrresrchlabbasic  gcr-singularity-lab     Singularity Lab          .                    V100
msrresrchvc        gcr-singularity-resrch  Singularity Shared       msra_srobot_azureml  A100, CPU, H100, MI200
palisades03        gcr-singularity-octo    Singularity Shared OCTO  .                    A100
quickdevvc         gcr-singularity-octo    Singularity Shared OCTO  .                    A100, MI300X
whitney03          gcr-singularity-octo    Singularity Shared OCTO  .                    MI300X
whitney09          gcr-singularity-octo    Singularity Shared OCTO  .                    MI300X
whitney13          gcr-singularity-octo    Singularity Shared OCTO  .                    MI300X
```
## 4. submit training
```bash
amlt run -t palisades03 amlt_tools/vln_A100.yaml # or any other cluster have A100s
```
1. after finish, you will get a training job link
2. you could modify scripts of `amlt_tools/vln_A100.yaml` if needed
