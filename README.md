# bag-of-poses

'Bag of Poses' is a group body language analysis tool. Combining the work of OpenPose - a real-time multi-person system to jointly detect human body, hand, and facial keypoints with a bag of words approach, 'Bag of Poses' is able to analyse large group images for supervised machine learning tasks.

This is a Python implementation with TensorFlow (CUDA optimised) and scikit-learn.

To download pre-trained Coco 4000 weights: `cd model; sh get_model.sh`


Project Architecture : 

	:::python 
	- root dir/
		|- images/
				|- test /
					|- obj1/
					|- obj2/

				|- train /
					|- obj1/
					|- obj2/

		|- helpers.py
		|- Bag.py 


	:~$ python Bag.py --train_path images/train/ --test_path images/test/

Saves checkpoint after each labelled object for loading large data:  `./checkpoint.pkl ` 

Remove completed labelled object by running again: 

`:~$ python Bag.py --train_path images/train/ --test_path images/test/ --checkpoint_path ./checkpoint.pkl`
