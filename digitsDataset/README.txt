Digits Dataset

1. Source
	-	MNIST dataset from Yann LeCun, Courant Institute, NYU
	-	Subset of the handwritten digits dataset from NIST (National Institute of Standards and Technology)

2. Sizes of dataset
	-	Training set: 6000 data points
	-	Validation set: 1000 data points
	-	Test set: 1000 data points

3. Features
	-	784 continuous real [0,1] attributes
		= normalized pixel values (each image is 28x28)
        (each pixel has an intensity from 0-255; we normalize and convert it to a real number in the range 0.0-1.0, then concatenate those 784 real numbers)

4. Format
	-	In each *Labels.csv, each line has one label from 0 through 9 representing the digit value.
	-	In each *Features.csv, each line represents a feature vector where each feature is separated by a comma.

5. Extra Information
    -   On the course web page we have provided an extra file with images of each of the digits.
