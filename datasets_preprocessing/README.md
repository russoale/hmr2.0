# Generating datasets for training and testing 

If you plan to train or evaluate HMR2.0, first you have to convert all datasets to tfrecords. 

The datasets used in this repository are the same used in [End-to-end Recovery of Human Shape and Pose](https://github.com/akanazawa/hmr/).

### Download datasets

2D datasets:
- [Leeds Sport Pose](https://sam.johnson.io/research/lsp.html)
- [Leeds Sports Pose Extended](https://sam.johnson.io/research/lspet.html)
- [MPII Human Pose](http://human-pose.mpi-inf.mpg.de)
- [MSCOCO](https://cocodataset.org/#home)

3D datasets:
- [MPII 3D Human Pose](http://gvv.mpi-inf.mpg.de/3dhp-dataset)
- [Human 3.6M](http://vision.imar.ro/human3.6m/description.php)
- [Total Capture](https://cvssp.org/data/totalcapture) (not used in the original paper)

SMPL Mesh datasets:
- follow the description in the [original repository](https://github.com/akanazawa/hmr/blob/master/doc/train.md#mosh-data)

If you use the datasets above, please cite the original papers and follow the individual license agreement.
  
### Generate TF records
```
Note, that the Human 3.6M and Total Capture datasets were available in a proprietary format!
If you plan to use the datasets, first change 
    - h36m_to_tfrecords.py
    - total_cap_to_tfrecords.py
to work with the original dataset formats!  
```
1. check `convert_datasets.sh` and change the paths
2. start virtual environment `workon hmr2.0`
3. run `convert_datasets.sh`


### Inspect data

Checkout the Jupyter notebooks in [src/visualise/notebooks](../src/visualise/notebooks)
1. Run `inspect_records.ipynb` 
2. Run `inspect_dataset.ipynb`
