## Installation

Create and activate environment.

```
conda create -n dela python=3.10
conda activate dela
```

Install pytorch.

```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Install ninja.

```
pip install ninja
```

Install the default llvm-openmp version 14.0.6.

```bash
conda install llvm-openmp
```

Install the default h5py version 3.9.0.

```
conda install h5py==3.9.0
pip install plyfile==1.0.2
pip install scipy==1.11.4
```

Install pointnet2_ops. 

```bash
cd utils/pointnet2_ops_lib/
pip install .
```



## Semantic Segmentation on S3DIS

### Dataset

```
Stanford3dDataset_v1.2_Aligned_Version.zip
```

Download Stanford3dDataset_v1.2_Aligned_Version dataset and unzip it. 

```
S3DIS
 |--- data
       |--- Stanford3dDataset_v1.2_Aligned_Version
             |--- Area_1
                   |--- conferenceRoom_1
                         |--- Annotations
                               |--- beam_1.txt
                               |--- wall_4.txt
                         |--- conferenceRoom_1.txt
             |--- Area_2
             |--- ...
             |--- Area_6
```

Then run **prepare_s3dis.py** to process raw data into tensors.

```
cd S3DIS
python prepare_s3dis.py
```

```
S3DIS
 |--- data
       |--- s3dis
             |--- 1_conferenceRoom_1.pt
             |--- ...
             |--- 6_pantry_1.pt
```



### Training and Test

Training or test.

```bash
cd S3DIS
python train.py

python test.py
```

```bash
cd S3DIS
python train.py
```

Modify the current idx in line 35 of **train.py**.

Saved logs and models are under **output/**. 

```
S3DIS
 |--- output
       |--- log
             |--- 01
                   |--- err.log
                   |--- out.log
       |--- model
             |--- 01
                   |--- best.pt
                   |--- last.pt
```




