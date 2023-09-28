package A2ML;

/**
 * @author:
 * @createDate: 
 * @description: A2ML is a algorithm which combines a short-term memory(SM) and a long-term memory(LM), and it is updated upon the arrival of each new example
 * The SM is a sliding windows, which always add new example at first, and delete the last one if it is full
 * The LM is a set of clusters. The description of the cluster is in the class of the cluster.
 *
 */


import MyTest.VectorOperators;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.*;
import moa.AbstractMOAObject;
import moa.classifiers.AbstractMultiLabelLearner;
import moa.core.Measurement;
import moa.core.SizeOf;

import java.util.ArrayList;
import java.util.Random;

public class A2ML extends AbstractMultiLabelLearner {
    public IntOption maxByteSizeOption = new IntOption("maxByteSize", 'm', "Maximum memory consumed by the tree.", 2147480000, 0, 2147483647);
    public IntOption RandomSeedOption = new IntOption("RandomSeed", 'd', "Seed for random.", 2);
    public IntOption windowSizeValue = new IntOption("WindowSize", 'w', "The maximum number of recent instances to store", 50);
    public IntOption rsOption = new IntOption("ReservoirSize", 's', "Size of reservoir sampling for each clusters",  100);
    public IntOption kOption = new IntOption("knn", 'k', "The k nearest neighbors for prediction",  3);

    private Random ran;
    private int numLab;
    private int numAtt;

    private int windowSize;


    private ArrayList<Instance> short_memory;
    private ArrayList<Cluster> long_memory;

    // Weights of SM and LM
    private ArrayList<Double> weight_SM;
    private ArrayList<Double> weight_LM;

    MultiLabelPrediction short_predict;
    MultiLabelPrediction long_predict;


    private int size_RS;
    private int knn;

    private int predictIndex = -1;
    private int[] predictIndices = null;


    private double[] attributeRangeMin;
    private double[] attributeRangeMax;

    @Override
    public void setModelContext(InstancesHeader context) {
        try {
            this.numLab = context.numOutputAttributes();
            this.numAtt = context.numInputAttributes();
            this.attributeRangeMin = new double[context.numInputAttributes()];
            this.attributeRangeMax = new double[context.numInputAttributes()];

            this.short_predict = new MultiLabelPrediction(this.numLab);
            this.long_predict = new MultiLabelPrediction(this.numLab);

        } catch(Exception e) {
            System.err.println("Error: no Model Context available.");
            e.printStackTrace();
            System.exit(1);
        }
    }

    @Override
    public void resetLearningImpl() {

        this.ran = new Random(RandomSeedOption.getValue());
        this.windowSize = this.windowSizeValue.getValue();
        this.long_memory = new ArrayList<Cluster>();

        this.short_memory = new ArrayList<Instance>();
        this.size_RS = this.rsOption.getValue();
        this.knn = this.kOption.getValue();
        this.weight_LM = new ArrayList<Double>();
        this.weight_SM = new ArrayList<Double>();
    }

    private void updateRanges(MultiLabelInstance instance) {
        for(int i = 0; i < instance.numInputAttributes(); i++) {
            if(instance.valueInputAttribute(i) < this.attributeRangeMin[i])
                this.attributeRangeMin[i] = instance.valueInputAttribute(i);
            if(instance.valueInputAttribute(i) > this.attributeRangeMax[i])
                this.attributeRangeMax[i] = instance.valueInputAttribute(i);
        }
    }


    public void initializeImpl() {
    }

    public int calcByteSize() {
        int size = (int) SizeOf.sizeOf(this);
        return size;
    }
    public int measureByteSize() {
        return this.calcByteSize();
    }

    @Override
    public void trainOnInstanceImpl(MultiLabelInstance multiLabelInstance) {
        updateRanges(multiLabelInstance);

        if (this.long_memory.isEmpty()){
            createNewCluster(multiLabelInstance);
        }else {
            double[] distances = getLabDistancesofCluster(this.long_memory, multiLabelInstance);
            int correctIndex = nArgMin(1, distances)[0];
            int actualModelSize = this.measureByteSize();
            if(distances[correctIndex] < 1.0 || actualModelSize > this.maxByteSizeOption.getValue() ){
                updateLongMemory(correctIndex, multiLabelInstance);
            }else {
                createNewCluster(multiLabelInstance);
            }
            updateHistories(multiLabelInstance);
        }

        // update short-term memory with FIFO strategy
        if(this.short_memory.size() == this.windowSize){
            this.short_memory.remove(0);
        }
        this.short_memory.add(multiLabelInstance);
    }

    /**
     * create a new cluster with the new example
     * @param multiLabelInstance The new example at time t
     */
    private void createNewCluster(MultiLabelInstance multiLabelInstance){
        Cluster newKernel = new Cluster(this);
        newKernel.setCluster(multiLabelInstance);
        this.long_memory.add(newKernel);
    }


    /**
     * Update long-term Memory with LVQ strategy
     * @param correctIndex The index of the best cluster
     * @param multiLabelInstance The new example at time t
     */
    private void updateLongMemory(int correctIndex, MultiLabelInstance multiLabelInstance){

        this.long_memory.get(correctIndex).updateCluster(multiLabelInstance);

        // if the prediction cluster is not the best cluster, their prototypes will be pulled apart
        if (this.predictIndex != -1 && correctIndex != this.predictIndex) {
            MultiLabelInstance pred = this.long_memory.get(this.predictIndex).getPrototype();
            pred = (MultiLabelInstance) VectorOperators.calculateCenter(pred, multiLabelInstance,this.size_RS, false);
            this.long_memory.get(this.predictIndex).setPrototype(pred);
            this.predictIndex = -1;
        }
    }


    /**
     * Returns the n indices of the smallest values (sorted).
     */
    private int[] nArgMin(int n, double[] distances) {

        int indices[] = new int[n];

        for (int i = 0; i < n; i++){
            double minValue = Double.MAX_VALUE;
            for (int j = 0; j < distances.length; j++){

                if (distances[j] < minValue){
                    boolean alreadyUsed = false;
                    for (int k = 0; k < i; k++){
                        if (indices[k] == j){
                            alreadyUsed = true;
                        }
                    }
                    if (!alreadyUsed){
                        indices[i] = j;
                        minValue = distances[j];
                    }
                }
            }
        }
        return indices;
    }

    /**
     * Computes the distance between vectors of labels.
     */
    private double[] getLabDistancesofCluster(ArrayList<Cluster>updateClusters, MultiLabelInstance multiLabelInstance) {

        double[] distances = new double[updateClusters.size()];
        for (int i = 0; i < updateClusters.size(); ++i) {
            distances[i] = VectorOperators.getCosLab(updateClusters.get(i).getPrototype(), multiLabelInstance);
        }
        return distances;
    }

    /**
     * Computes the distance between vectors of feature.
     */
    private double[] getAttDistancesofCluster(ArrayList<Cluster>predictClusters, MultiLabelInstance multiLabelInstance) {
        double[] distances = new double[predictClusters.size()];
        for (int i = 0; i < predictClusters.size(); ++i){
            MultiLabelInstance centroids = predictClusters.get(i).getPrototype();
            distances[i] = VectorOperators.getCosAtt(centroids, multiLabelInstance, this.attributeRangeMax, this.attributeRangeMin);
        }
        return distances;
    }


    /**
     * The prediction is derived from an aggregation of two prediction candidates, one of STM, the other of LTM.
     * And the contributions of STM and LTM prediction candidates depend on their weights, which quantify the accuracy of STM and LTM.
     *      The LTM prediction is determined in the reservoir of the predict cluster with the knn classification.
     *
     * @param multiLabelInstance the new example at time t
     * @return the final prediction
     */
    @Override
    public Prediction getPredictionForInstance(MultiLabelInstance multiLabelInstance) {
        MultiLabelPrediction prediction = new MultiLabelPrediction(this.numLab);
        if(!this.long_memory.isEmpty() && this.weight_LM.size() > 0){
            double weightST = 1.0D;
            double weightLT = 1.0D;
            for (int i = 0; i < this.weight_LM.size(); i++) {
                weightST += this.weight_SM.get(i);
                weightLT += this.weight_LM.get(i);
            }

            double sumWeight = weightST + weightLT;
            weightST = weightST/sumWeight;
            weightLT = weightLT/sumWeight;

            double[] distances = getAttDistancesofCluster(this.long_memory, multiLabelInstance);
            this.predictIndex = nArgMin(1, distances)[0];
            this.long_predict = (MultiLabelPrediction) this.long_memory.get(this.predictIndex).getPredictionForInstance(multiLabelInstance);
            this.short_predict = getSTMPrediction(multiLabelInstance);

            for(int j = 0; j < this.numLab; j++) {
                double count = 0;
                count += weightLT * this.long_predict.getVote(j,1);
                count += weightST* this.short_predict.getVote(j,1);
                prediction.setVotes(j, new double[]{1.0 - count, count});
            }

        }else {

            prediction = getSTMPrediction(multiLabelInstance);
        }
        return prediction;
    }

    /***
     * The knn and majority vote classification in STM
     * @param multiLabelInstance the new example at time t
     * @return the STM prediction
     */
    protected MultiLabelPrediction getSTMPrediction(MultiLabelInstance multiLabelInstance){
        MultiLabelPrediction prediction = new MultiLabelPrediction(this.numLab);
        double[] distances = new double[this.short_memory.size()];

        for (int i = 0; i < this.short_memory.size(); ++i) {
            distances[i] = VectorOperators.getCosAtt(this.short_memory.get(i),multiLabelInstance, this.attributeRangeMax,this.attributeRangeMin);
        }

        int[] nnIndices = nArgMin(Math.min(distances.length, this.knn), distances);
        for(int j = 0; j < this.numLab; j++) {

            int count = 0;
            for (int nnIdx : nnIndices){
                if (this.short_memory.get(nnIdx).classValue(j) == 1)
                    count++;
            }
            double relativeFrequency = count / (double) (this.knn);
            prediction.setVotes(j, new double[]{1.0 - relativeFrequency, relativeFrequency});
        }
        return prediction;
    }

    /***
     * The weight of STM/lTM is the mean of the accuracy between the prediction candidates and the true label vectors over the last m examples,
     * where m is the size of STM.
     */
    protected void updateHistories(MultiLabelInstance multiLabelInstance) {
        double shortAcc = AccPred(this.short_predict, multiLabelInstance);
        if (this.weight_SM.size() == this.windowSize) {
            this.weight_SM.remove(0);
        }
        this.weight_SM.add(shortAcc);

        double longAcc = AccPred(this.long_predict, multiLabelInstance);
        if (this.weight_LM.size() == this.windowSize) {
            this.weight_LM.remove(0);
        }
        this.weight_LM.add(longAcc);
    }

    public static double AccPred(MultiLabelPrediction pred, Instance instance) {
        double cur_tp = 0;
        double cur_fp = 0;
        double cur_fn = 0;
        double delta_exAcc = 0.0;
        for(int i = 0; i < pred.size(); i++) {
            double yp = (pred.getVote(i,1) >= 0.5) ? 1 : 0;
            double yt = instance.classValue(i);
            cur_tp   += (yt == 1 && yp == 1) ? 1 : 0;
            cur_fn   += (yt == 1 && yp == 0) ? 1 : 0;
            cur_fp   += (yt == 0 && yp == 1) ? 1 : 0;

        }

        if(cur_tp + cur_fn + cur_fp > 0)
        {
            delta_exAcc = cur_tp / (cur_tp + cur_fn + cur_fp);
        }
        return delta_exAcc;
    }


    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[0];
    }

    @Override
    public void getModelDescription(StringBuilder stringBuilder, int i) {

    }

    @Override
    public boolean isRandomizable() {
        return false;
    }


    /***
     * The long-term Memory consist of s clusters.
     * Each cluster is a triplet made up of a pair of two prototypes (X-prototype and Y-prototype) and a reservoir
     * -    Each pair of prototypes will be updated by LVQ strategy
     * -    Each reservoir is updated by Biased Reservoir Sampling
     */

    public static class Cluster extends AbstractMOAObject {
        protected A2ML LTM;

        protected MultiLabelInstance prototypes;
        private ArrayList<MultiLabelInstance> reservoir;


        public Cluster(A2ML LTM) {
            this.LTM = LTM;
            this.reservoir = new ArrayList<MultiLabelInstance>();
        }

        // Initialiser le cluster
        private void setCluster(MultiLabelInstance instance){
            setPrototype(instance);
            this.reservoir.add(instance);
        }
        public void setPrototype(MultiLabelInstance instance) {
            this.prototypes = instance;
        }

        public MultiLabelInstance getPrototype() {
            return prototypes;
        }


        /**
         * Store new instances in the reservoir of the best Cluster by biased reservoir sampling rules
         * and update the prototypes with LVQ strategy
         * @param multiLabelInstance the new example at time t
         */
        private void updateCluster(MultiLabelInstance multiLabelInstance) {
            this.prototypes = (MultiLabelInstance) VectorOperators.calculateCenter(this.prototypes, multiLabelInstance, this.reservoir.size(), true);
            if (this.reservoir.size() < this.LTM.size_RS) {
                this.reservoir.add(multiLabelInstance);
            } else{
                int replace = this.LTM.ran.nextInt(this.LTM.size_RS);
                this.reservoir.set(replace, multiLabelInstance);
            }
        }

        /**
         * The prediction in the predict cluster with a knn and majority vote classification
         * @param multiLabelInstance the new example at time t
         * @return the prediction
         */
        public Prediction getPredictionForInstance(MultiLabelInstance multiLabelInstance) {

            MultiLabelPrediction prediction = new MultiLabelPrediction(this.LTM.numLab);

            double[] preDistance;
            preDistance = get1ToNAttDistances(multiLabelInstance);

            this.LTM.predictIndices = this.LTM.nArgMin(Math.min(preDistance.length, this.LTM.knn), preDistance);

            for(int j = 0; j < this.LTM.numLab; j++) {

                int count = 0;
                for (int nnIdx : this.LTM.predictIndices){
                    if (this.reservoir.get(nnIdx).classValue(j) == 1)
                        count++;
                }
                double relativeFrequency = count / (double) (this.LTM.knn);
                prediction.setVotes(j, new double[]{1.0 - relativeFrequency, relativeFrequency});
            }
            return prediction;
        }

        /**
         * Computes the distance between a vector of feature and a list of vectors of features.
         */
        private double[] get1ToNAttDistances(MultiLabelInstance multiLabelInstance) {
            double[] distances = new double[this.reservoir.size()];
            for (int i = 0; i < this.reservoir.size(); ++i) {
                distances[i] = VectorOperators.getCosAtt(this.reservoir.get(i), multiLabelInstance, this.LTM.attributeRangeMax,this.LTM.attributeRangeMin);
            }
            return distances;
        }

        public void getDescription(StringBuilder sb, int indent) {
        }

    }
}
