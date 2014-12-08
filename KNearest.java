/** K Nearest Neighbors for CS170 Fall 2014
    Multi-threaded 
    by Evan Fossier
**/

import java.util.ArrayList;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.Math;
import java.lang.Thread;
import java.lang.Runnable;

public class KNearest{
    public final int K_SIZE;
    public final int VECTOR_SIZE;

    public int numTrainingPoints = 6000; // Initial estimate for number of training points, recalculated when parsing training file
    public int numOutputPoints = 1000; //Initial estimate, recalculated when parsing the test file

    public boolean trained = false;

    public int[] trainingLabels;
    public ArrayList<double[]> trainingVectors; 

    public ArrayList<Integer> outputLabels;
    public ArrayList<double[]> outputVectors;

    /**
     * Initializes the Knearest neighbor classifier by reading in the training data from the given file
     * 
     * @param  k                number of neighbors to look at
     * @param  vectorSize       size of each vector in terms of number of features
     * @param  trainingVectors  the name of the file containing the training features
     * @param  trainingLabels   the name of the file containing the training labels
     */
    public KNearest(int k, int vectorSize, String trainingVectors, String trainingLabels) throws FileNotFoundException,IOException{
        K_SIZE = k;
        VECTOR_SIZE = vectorSize;
        System.out.println("Initializing K-Nearest Neighbors Classifier");
        System.out.println("K: "+k);
        System.out.println("Vector size: "+vectorSize);
        System.out.println("Training Vector File: "+trainingVectors);
        System.out.println("Training Labels File: "+trainingLabels);
        System.out.println("");

        parseTrainingVectors(trainingVectors);
        System.out.println("");
        parseTrainingLabels(trainingLabels);

        trained = true;
        System.out.println("");
    }

    /**
     * Reads the file pointed to by trainingVectors and stores the vectors in the ArrayList trainingVectors
     * @param trainingVectors filename of file containing the training vectors
     */
    private void parseTrainingVectors(String trainingVectors) throws FileNotFoundException,IOException{
        BufferedReader br = new BufferedReader(new FileReader(trainingVectors));
        System.out.println("Parsing training vectors from "+trainingVectors);

        String vector = "";
        String csvSplitOn = ","; //split on comma for csvs
        int actualNumVectors = 0;

        int progressIncrement = numTrainingPoints/10; //Want to report in increments of 10%

        // Init the training vectors array
        this.trainingVectors = new ArrayList<double[]>(numTrainingPoints);

        // Read the file
        System.out.print("|");
        while((vector = br.readLine()) != null){
            // Split on commas
            String[] features = vector.split(csvSplitOn);

            // Ensure each vector has same size
            if(features.length != VECTOR_SIZE){
                System.out.println("Incorrect vector size on vector "+(actualNumVectors+1));
                br.close();
                return;
            }
            // Parse each feature in the vector and put it in corresponding vector array
            double[] vectorVals = new double[VECTOR_SIZE];
            for(int i=0; i<VECTOR_SIZE; i++){
                vectorVals[i] = Double.parseDouble(features[i]);
            }
            this.trainingVectors.add(vectorVals);
            actualNumVectors++;

            // Report progress
            if((actualNumVectors % progressIncrement) == 0){
                int amount = (actualNumVectors / progressIncrement)*10;
                System.out.print(" "+amount+"% ");
            }
        }
        System.out.print("|");
        System.out.println("\nTotal vectors processed: "+actualNumVectors);
        numTrainingPoints = actualNumVectors;

        br.close();
    }

    /**
     * Reads the file pointed to by trainingLabels and stores the labels in the trainingLabels array
     * @param trainingLabels filename
     */
    private void parseTrainingLabels(String trainingLabels) throws FileNotFoundException,IOException{
        BufferedReader br = new BufferedReader(new FileReader(trainingLabels));
        System.out.println("Parsing training labels from "+trainingLabels);

        // Init label array
        this.trainingLabels = new int[numTrainingPoints];

        String label = "";
        int actualNumLabels = 0;
        int progressIncrement = numTrainingPoints/10; //Want to report in increments of 10%

        // read the file
        System.out.print("|");
        while((label = br.readLine()) != null){
            // make sure we dont go out of bounds of the array
            if(actualNumLabels == numTrainingPoints){
                System.out.println("\nError: Too many data points in training labels file");
                br.close();
                return;
            }
            // Parse the label into an int
            this.trainingLabels[actualNumLabels] = Integer.parseInt(label);

            actualNumLabels++;
            // Report progress
            if((actualNumLabels % progressIncrement) == 0){
                int amount = (actualNumLabels / progressIncrement) * 10;
                System.out.print(" "+amount+"% ");
            }
        }
        System.out.print("|");
        System.out.println("\nTotal labels processed: "+actualNumLabels);
        if(actualNumLabels != numTrainingPoints){
            System.out.println("Error: Number of labels doesn't match up to number of training data points!");
        }

        br.close();
    }

    /**
     * Returns wether kNearest has been trained, ie ready to proceed to classification
     * @return true if Knearest neighbors has been trained
     */
    public boolean isTrained(){
        return trained;
    }

    /**
     * A worker thread to classify a portion of the input points
     */
    class ClassifyThread extends Thread{

        private int startIdx;
        private int endIdx;
        private int tnum;

        public ClassifyThread(int tnum, int start, int end){
            startIdx = start;
            endIdx = end;
            this.tnum = tnum;
            System.out.println("Thread "+tnum+" assigned indices: "+start+" to "+end);
        }

        @Override public void run(){
            System.out.println("Thread "+tnum+" started");
            classifyThread(startIdx, endIdx);
            System.out.println("Thread "+tnum+" finished");
        }
    }

    /**
     * Opens the input file and loads all of the input data points into memory, then splits work across threads and launches classification
     * @param  inputVectors          the file containing all of the input vectors to classify
     * @throws FileNotFoundException 
     * @throws IOException           
     * @throws InterruptedException  
     */
    public void classify(String inputVectors) throws FileNotFoundException, IOException, InterruptedException{
        BufferedReader br = new BufferedReader(new FileReader(inputVectors));
        System.out.println("Classifying datapoints from "+inputVectors);

        String vector;
        this.outputVectors = new ArrayList<double[]>(numOutputPoints);
        System.out.println("Reading vectors from file");
        while((vector = br.readLine()) != null){
            String[] v = vector.split(",");
            double[] vVals = new double[VECTOR_SIZE];
            for(int i=0; i<v.length; i++){
                vVals[i] = Double.parseDouble(v[i]);
            }
            this.outputVectors.add(vVals);
        }

        // Setting up output array
        System.out.println("Setting up output array");
        this.outputLabels = new ArrayList<Integer>(numOutputPoints);
        for(int i=0; i<numOutputPoints; i++){
            this.outputLabels.add(-1);
        }
        System.out.println("Setting up threads");
        int NUM_THREADS = 8;
        ClassifyThread[] allThreads = new ClassifyThread[NUM_THREADS];
        int incr = numOutputPoints/NUM_THREADS;
        int start = 0;
        int end = incr;
        for(int j=0; j<NUM_THREADS; j++){
            allThreads[j] = new ClassifyThread(j, start, end);
            start = end;
            end  = end + incr;
        }
        System.out.println("Starting threads");
        // start each thread
        for(int k =0; k<NUM_THREADS; k++){
            allThreads[k].start();
        }

        // wait for each thread to finish
        for(int k=0; k<NUM_THREADS; k++){
            allThreads[k].join();
        }

        System.out.println("All threads finished");
    }

    /**
     * The thread level classify, each thread classifies outputVectors[startIdx:endIdx] (endIdx excluded) and writes the results to outputLabels[startIdx:endIdx]
     * @param startIdx the start index
     * @param endIdx   the end index
     */
    public void classifyThread(int startIdx, int endIdx){
        for(int pointNum=startIdx; pointNum<endIdx; pointNum++){
            System.out.println("Classifying point "+pointNum);
            double[] vectorVals = this.outputVectors.get(pointNum);

            int[] nearestJonNeighbors = nearestNeighbors(vectorVals); // indices of the closest neighbors

            // get the labels for those neighbors
            int[] neighborLabels = new int[K_SIZE];
            for(int j=0; j<K_SIZE; j++){
                neighborLabels[j] = trainingLabels[nearestJonNeighbors[j]];
            }
            // Find the majority label
            int majorityLabel = majorityElem(neighborLabels);

            // Check if there was a majority element
            if(majorityLabel == -1){
                System.out.println("Point: "+pointNum+" No majority label elem for this item");
                // Pick the closest point to this one
                double bestDist = Double.MAX_VALUE;
                int bestIdx = 0;
                for(int i=0; i<K_SIZE; i++){
                    int pointIdx = nearestJonNeighbors[i]; //the actual index of the training vector
                    double d = euclidianDist(vectorVals, trainingVectors.get(pointIdx));
                    if(d < bestDist){
                        bestDist = d;
                        bestIdx = pointIdx;
                    }
                }
                this.outputLabels.set(pointNum, this.trainingLabels[bestIdx]);
                System.out.println(">>Point "+pointNum+" classified to label "+this.trainingLabels[bestIdx]);
            }else{
                this.outputLabels.set(pointNum, majorityLabel);
                System.out.println(">>Point "+pointNum+" classified to label "+majorityLabel);
            }
        }
    }

    /**
     * Finds the majority element in array
     * @param  array the array of values to find the majority elem in
     * @return       the majority element or -1 if not majority element
     */
    private int majorityElem(int[] array){
        int count = 0, majorityElement = 0;
        for (int i = 0; i < array.length; i++) {
            if (count == 0)
                majorityElement = array[i];
            if (array[i] == majorityElement) 
                count++;
            else
                count--;
        }
        count = 0;
        for (int i = 0; i < array.length; i++)
            if (array[i] == majorityElement)
                count++;
        if (count > array.length/2)
            return majorityElement;
        return -1;

    }

    /**
     * Finds the k nearest neighbors by euclidian distance from the feature vector among the training points
     * @param  featureVector the point we are trying to find neighbors of
     * @return               an ArrayList of the indices of the nearest neighbors
     */
    private int[] nearestNeighbors(double[] featureVector){
        int[] result = new int[K_SIZE]; // The indicies of the k closest training points to the nearest neighbor
        double[] bestDistances = new double[K_SIZE]; // The k best distances from featureVector
        double largestDist = Double.MAX_VALUE; // The largest dist out off our k best distances

        // init bestDistances, result
        for(int i=0; i<K_SIZE; i++){
            result[i] = -1;
            bestDistances[i] = Double.MAX_VALUE;
        }

        // walkthrough all the training vectors and calculate euclidian distance
        for(int i=0; i<numTrainingPoints; i++){
            double d = euclidianDist(featureVector, trainingVectors.get(i));
            // We can replce one of the closest points in bestDistances
            if(d < largestDist){
                // Find the largest elem in bestDistances to replace
                int largestIdx = 0;
                double largest = 0;
                for(int j=0; j<K_SIZE; j++){
                    if(bestDistances[j] > largest){
                        largestIdx = j;
                        largest = bestDistances[j];
                    }
                }
                // Replace the largest elem with the newly calculated distance
                result[largestIdx] = i;
                bestDistances[largestIdx] = d;
                // Find the new largest elem
                largest = 0;
                for(int j=0; j<K_SIZE; j++){
                    if(bestDistances[j] > largest){
                        largest = bestDistances[j];
                    }
                }
                largestDist = largest;
            }
        }

        return result;    
    }

    /**
     * Calculates the euclidian distance between v1 and v2
     * @param  v1 source vector
     * @param  v2 destination vector
     * @return    the euclidian distance between v1 and v2.
     */
    private double euclidianDist(double[] v1, double[] v2){
        double result = 0.0;
        for(int i=0; i<VECTOR_SIZE; i++){
            result = result + Math.pow(v1[i] - v2[i], 2.0);
        }

        return Math.sqrt(result);
    }

    /**
     * Writes the contents of outputLabels array to outputFileName in CSV format
     * @param outputFileName what to name the output file
     */
    public void outputClassification(String outputFileName) throws IOException{
        System.out.println("Outputting classification to "+outputFileName);
        BufferedWriter bw = new BufferedWriter(new FileWriter(outputFileName));

        for(int i=0; i<numOutputPoints; i++){
            String v = outputLabels.get(i).toString();
            bw.write(v);
            bw.newLine();
        }

        System.out.println("Done outputing labels");
        bw.close();
    }

    /**
     * Compares the resulting classification with known correct labels and reports the percentage correct
     * @param  correctOutputName file containing the correct output labels
     * @param  reportNum         will print both vectors on an incorrect match to file missedPoints.txt for the first 'reportNum' mistakes
     */
    public void checkOutput(String correctOutputName, int reportNum) throws FileNotFoundException, IOException{
        if(outputLabels.size() == 0){
            System.out.println("No output to compare with");
            return;
        }
        BufferedReader br = new BufferedReader(new FileReader(correctOutputName));
        System.out.println("Comparing output labels with correct labels in "+correctOutputName);

        BufferedWriter bw;
        if(reportNum > 0)
            bw = new BufferedWriter(new FileWriter("missedPoints.txt"));
        else bw = null;

        int i = 0;
        int numErrors = 0;
        String line = "";

        while(((line = br.readLine()) != null) && i < numOutputPoints){
            int correctLabel = Integer.parseInt(line);
            if(correctLabel != outputLabels.get(i)){
                numErrors++;
                if(reportNum >0 && numErrors <= reportNum){
                    bw.write("Error: mismatch on vector " +i + ", was " + outputLabels.get(i) + ", should be " + correctLabel);
                    bw.newLine();
                    bw.write("Vector: <");
                    double[] badPoint = this.outputVectors.get(i);
                    for(int j=0; j<VECTOR_SIZE; j++){
                        bw.write(String.valueOf(badPoint[j]));
                        if(j!=(VECTOR_SIZE-1)){
                            bw.write(", ");
                        }
                    }
                    bw.write(">");
                    bw.newLine();
                    bw.newLine();
                }
            }
            i++;
        }

        if(i < numOutputPoints){
            System.out.println("Error: Not enough points in the correct file!");
        }
        else if(line != null){
            System.out.println("Error: More points in the correct file then in classified data!");
        }
        br.close();
        if(reportNum > 0)
            bw.close();
        double acc = ((double)(numOutputPoints-numErrors))/numOutputPoints;
        System.out.println("Accuracy: "+(numOutputPoints-numErrors)+" out of "+numOutputPoints+", "+ (acc*100) +"%");
    }

    /**
     * Driver program which takes args from command line and runs Knearest neighbors
     * @param args[] [description]
     */
    public static void main(String[] args){
        if(args.length < 2){
            System.out.println("Usage: java KNearest k TestVectorsFilename [testLabelsFilename]");
            System.exit(1);
        }

        int k = Integer.parseInt(args[0]);
        int vectorSize = 28*28;
        String trainingVectorsFileName = "digitsDataset/trainFeatures.csv";
        String trainingLabelsFileName = "digitsDataset/trainLabels.csv";
        String testVectorsFileName = args[1];
        String testLabelsFileName = "";
        boolean checkOutput = false;
        if(args.length > 2){
            checkOutput = true;
            testLabelsFileName = args[2];
        }

        KNearest classifier;

        try{
            long startTime = System.currentTimeMillis();
            classifier = new KNearest(k, vectorSize, trainingVectorsFileName, trainingLabelsFileName);
            classifier.classify(testVectorsFileName);
            if(checkOutput){
                classifier.checkOutput(testLabelsFileName, 0);
            }else{
                classifier.outputClassification("KNeighborsOut.csv");
            }
            long endTime = System.currentTimeMillis();
            long total = endTime - startTime;
            total = total/1000;
            System.out.println("Total runtime: "+total+" s");
        }
        catch(FileNotFoundException fe){
            System.out.println("Could not find one of the input files");
        }
        catch(IOException ie){
            System.out.println("Error reading one of the input files");
        }
        catch(InterruptedException iie){
            System.out.println("Thread interrupted");
        }

    }

}