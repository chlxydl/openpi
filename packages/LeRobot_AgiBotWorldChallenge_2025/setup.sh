cd /work
mkdir -p ~/.blobfuse2
echo $AZURE_ACCESS_TOKEN
echo "test"
/work/amlt_tools/aml/mount_storage.sh
pip install -e ".[test, aloha, xarm, pusht, dynamixel, smolvla]"
pip install -U amlt --index-url https://msrpypi.azurewebsites.net/stable/leloojoo
/bin/bash


scp -i "C:/Users/v-yuboguo/keys/id_rsa" -r "C:\Users\v-yuboguo\new_sim_data_0813.zip" v-yuboguo@microsoft.com@gcrazgdl1528:/home/v-yuboguo/Datasets
