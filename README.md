# AI/ML Accelarator
## Aim
Our aim is to reduce the no of calculations in vision transformer model by removing the unnecessary multiplication of same pairs(weights) in a matrix.

To increase the chances of finding same pairs we will round off all the weights to a particular decimal place. Doing so will result in increase of percentage hits(percentage of same pairs) but will also reduce the accuracy of model.

We will compute both accuracy loss and percentage hits for 3 different models of ViT. Both the things can also be computed for other models of ViT using same code with very minor changes. 
## Files
There are two seperate files to find the accuracy loss and percentage hits after tempering the weights.
   1. Accuracy Loss
   2. Percentage Hits
### Accuracy Loss
After importing the necessary classes, we will temper the weights of **Multi-Head Attention Layer(MHA)** and **linear layers** of **MLP** block by rounding them off to a given decimal place.

Now we will calculate the accuracy for the original weights and then find the loss in accuracy with the accuracy of model with tempered weights.

While tempering weights, we have approximately tempered 24/25th of the total calculation, in [MHA](https://github.com/pytorch/pytorch/blob/e3eb1d92d8e26db37a0c06e40b71d744b7a5fc63/aten/src/ATen/native/transformers/attention.cpp#L223) layer `348,585,984` calculation is happening on line 299 which has two matricies `query` of dimension `1×197×768` and `qkv_weight` matrix of dimension `768×2304` , on line 373 which consists of `348,585,984` calculation has two matrices `attention_ctx` of dimension `197×768` and `projection_weight` of dimension `768 × 2304` and we have tempered these calculations in MHA layer.

The calculations which we have not tempered in [MHA](https://github.com/pytorch/pytorch/blob/e3eb1d92d8e26db37a0c06e40b71d744b7a5fc63/aten/src/ATen/native/transformers/attention.cpp#L223) are give on line 340 which has two matrice `q` of dimension `1×197×768` and `k` of dimension `1×197×768` and it consists of `29,805,312` calculations, line 359 consists of `29,805,312` calculations which has two matricies `q` of dimension `1×197×768` and `qkt` of dimension `197×197`.

In MLP layer both the linear layers consists of `464,781,312` calculations and we have tempered both the layers. The dimensions of two matricies `A` and `B` in linear layers are `197×768` and `768×3072`.

Reference code for all the above mentioned layers can be found in the publicly available code of [ViT](https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py#L16) and cpp file of [MHA](https://github.com/pytorch/pytorch/blob/e3eb1d92d8e26db37a0c06e40b71d744b7a5fc63/aten/src/ATen/native/transformers/attention.cpp#L223) layer can also be used for reference. 

Data for the accuracy loss of 3 models namely ViT B-16, ViT B-32 and ViT L-16 can be found in the attached excel sheet and exact commented code for ViT B-16 is also attached. Same code can be used for other ViT models also by just changing the name of the model in line 32,36 and 37 of the code.

---
### Percentage Hits
To find the percentage of same pairs (pairs of weights in a matrix, basically one pair corresponds to one calculation thus it implies that removing 50% pairs also reduces the calculation by 50%) we will round them off to a particular decimal place and then identify the same pairs.

By same pairs we are considering all the below given cases as same pairs:

1. (1.2,2.3)
2. (2.3,1.2)
3. (-1.2,2.3)
4. (2.3,-1.2)
5. (1.2,-2.3)
6. (-2.3,1.2)
7. (-1.2,-2.3)
8. (-2.3,-1.2)

In order to consider all the above cases, first of all we will take mod of every weight thus removing negative sign from every case. This will leave us with only two cases (1,2) and (2,1).

Now first of all we will create a set and add all the pairs in it by converting them into a string, as set consists of only unique pairs, exactly same pairs like (1,2) and (1,2) will be added only once.

To remove a reverse pair we will remove the string, reverse it and again add it into the set. Now if the reverse part of that string was already present the current string will not be added and if it was not present it will be added (as it should be because we initially removed it and it should be present for once at least).

> Due to the shortage of resources, in the attached data we have analyzed only 12 lakh pairs to calculate the percentage hits (to calculate the accuracy we are rounding off all the weights as above mentioned because it is not time consuming) but to maintain the accuracy of our analysis we have created the concept of row and column zones, and row and column population in which we will divide whole matrix into given row and column zones and pick some random elements according to the row and coulmn population thus maintaining the accuracy of our analysis.
>
> Furthur the attached data is the average of data of 10 images, running each image for 3 times. (As we are doing random sampling, answer will vary on each run but with a very minor difference).

## Conclusion
By increasing the no of decimal places to be rounded off, we can increase the percentage hits (can be proven easily using mathematics) but we will also reduce the accuracy of our model.

From our data we have come to the conclusion that rounding off to 2 decimal places (where accuracy loss is 2-4% and percentage hits are 99.6-99.8%) or 3 decimal places (where accuracy loss is 0.2-0.4% and percentage hits are 90-95%) are the two best scenerios.
