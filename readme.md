ORIN NANO
orin
ssh orin@192.168.50.135
sudo docker run --runtime nvidia -it --rm -v /run/jtop.sock:/run/jtop.sock --network=host dustynv/l4t-pytorch:r36.4.0
sudo docker exec -it 7b6a bash

Jetson NANO
enigma123
ssh newcastleuni@192.168.50.94
sudo docker run --runtime nvidia -it --rm -v /run/jtop.sock:/run/jtop.sock --network=host dustynv/l4t-pytorch:r32.7.1
sudo docker exec -it 7b6a bash

scp -r images newcastleuni@192.168.50.94:nvdli-data
scp imagenet-classes.txt newcastleuni@192.168.50.94:nvdli-data
sudo docker cp imagenet-classes.txt 8a84:/home


Raspberry Pi
ssh pi@192.168.50.203
raspberry
