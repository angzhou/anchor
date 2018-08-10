## ANCHOR

**08/2018 update:** Our single model top-1 test accuracy is now 97.68%.  We changed the model a little bit and also applied a few optimization techniques.  As of this time (08/2018), our result is again the best in the world, i.e., the state of the art. (end of update)

---

Here are model, weights, and test samples for our handwritten Chinese character recognition(HCCR) development.

The model is VGG like, we did try other CNN architectures such as ResNet, DenseNet, but so far this VGG-like model performs the best.  We also tried `wider` (increase number of conv filters per layer) nets, their results is on par with smaller one but take much longer to train.

As of now (04/2017), this model achieves better test accuracy (97.25%) than any other published results, including ensemble results. (With ensemble, we did actually have even better accuracy!).

The test data provided are a subset (about 1/10th) of test data from ICDAR 2013 Chinese Handwriting Recognition Competition. There are 3755 characters, each character has 6 samples.  So there are 3755*5=22530 samples total. 

(The data is copyrighted by Institute of Automation, Chinese Academy of Science, see: C.-L. Liu, F. Yin, D.-H. Wang, Q.-F. Wang, CASIA online and offline Chinese handwriting databases, Proc. 11th International Conference on Document Analysis and Recognition (ICDAR), Beijing, China, 2011, pp.37-41.)

## Install

Just do:

    pip install keras tensorflow-gpu h5py pillow

If you don't have GPU, replace `tensorflow-gpu` with `tensorflow`, note it takes significantly longer to run without GPU.

## Usage

Clone this repo. (Warning: this is a big repo >100M, clone with caution.)

If no argument:

    python model_test.py

will run all test samples in `data/test` (22530 total) and will print out loss and accuracy.

If provided `-p` argument:

    python model_test.py -p

It will give 3 top predictions for those samples that the model gives wrong first prediction.

If you give it a filename:

    python model_test.py sample.png

It will print 3 top predictions for the single sample.

## Fun

![峻 C054-f-f.png](/data/test/峻/C054-f-f.png) Model is 100% sure it is `峨`, unfortunately it is `峻`.

![拨 C044-f-f.png](/data/test/拨/C044-f-f.png) Our model is 100% sure it is `拔`, but it is `拨`

![尹 C017-f-f.png](/data/test/尹/C017-f-f.png) model is 100 sure it is `君`, but it is `尹`.

![挚 C016-f-f.png](/data/test/挚/C016-f-f.png) Model is 100% sure it is `热`, but it is `挚`.

AFAICT, model is correct in all these cases, test data are simply incorrectly labeled. (Or the writer wrote the wrong characters)

Now for some unsure predictions by the model:

![雀 C046-f-f.png](/data/test/雀/C046-f-f.png) Model is about 50/50 on `崔` and `雀`, I am too. (labeled `雀`)

![卡 C017-f-f.png](/data/test/卡/C017-f-f.png) Model is 50/50 on `长` and `卡`, so am I. (labeled `卡`)

![荡 C044-f-f.png](/data/test/荡/C044-f-f.png) Model is torn between `荷` and `荡`. (labeled `荡`)

![队 C017-f-f.png](/data/test/队/C017-f-f.png) Top 3 are `义戏叉`, it's messed up. (labeled `队`)


### Name

ANCHOR stands for "ANgzhou Chinese Handwriting Optical Recognition" 
