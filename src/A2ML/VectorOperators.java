package A2ML;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import com.yahoo.labs.samoa.instances.SparseInstance;

import java.util.ArrayList;


public class VectorOperators {


    /**
     * Cosine Similarity  of feature vectors
     */
    public static double getCosAtt(Instance instance1, Instance instance2, double[] attributeRangeMax, double[] attributeRangeMin ) {

        double distance = 0.0D;
        double distanceA = 0.0D;
        double distanceB = 0.0D;

        if(instance1.numValues() == instance1.numAttributes()) // Dense Instance
        {
            for(int i = 0; i < instance1.numInputAttributes(); i++)
            {
                double val1 = instance1.valueInputAttribute(i);
                double val2 = instance2.valueInputAttribute(i);

                if(attributeRangeMax[i] - attributeRangeMin[i] != 0)
                {
                    val1 = (val1 - attributeRangeMin[i]) / (attributeRangeMax[i] - attributeRangeMin[i]);
                    val2 = (val2 - attributeRangeMin[i]) / (attributeRangeMax[i] - attributeRangeMin[i]);
                }
                distance += val1 * val2;
                distanceA += val1 * val1;
                distanceB += val2 * val2;
            }
        }
        else // Sparse Instance
        {

            int firstI = -1, secondI = -1;
            int firstNumValues  = instance1.numValues();
            int secondNumValues = instance2.numValues();
            int numAttributes   = instance1.numAttributes();
            int numOutputs      = instance1.numOutputAttributes();

            for (int p1 = 0, p2 = 0; p1 < firstNumValues || p2 < secondNumValues;) {

                if (p1 >= firstNumValues) {
                    firstI = numAttributes;
                } else {
                    firstI = instance1.index(p1);
                }

                if (p2 >= secondNumValues) {
                    secondI = numAttributes;
                } else {
                    secondI = instance2.index(p2);
                }

                int idx1, idx2;

                if (instance1.classIndex() == 0) {
                    idx1 = firstI - numOutputs;
                    idx2 = secondI - numOutputs;
                }else{
                    idx1 = firstI;
                    idx2 = secondI;
                }

                if (instance1.classIndex() == 0) {
                    if (firstI < numOutputs) {
                        p1++;
                        continue;
                    }
                    if (secondI < numOutputs) {
                        p2++;
                        continue;
                    }
                }else{
                    if (firstI >= instance1.classIndex() && secondI >= instance1.classIndex()) {
                        break;
                    }
                }

                if(firstI == secondI) {

                    double val1 = instance1.valueSparse(p1);
                    double val2 = instance2.valueSparse(p2);
                    if(attributeRangeMax[idx1] - attributeRangeMin[idx1] != 0) {
                        val1 = (val1 - attributeRangeMin[idx1]) / (attributeRangeMax[idx1] - attributeRangeMin[idx1]);
                        val2 = (val2 - attributeRangeMin[idx1]) / (attributeRangeMax[idx1] - attributeRangeMin[idx1]);
                    }
                    distance += val1 * val2;
                    distanceA += val1 * val1;
                    distanceB += val2 * val2;
                    p1++;
                    p2++;

                } else if (firstI > secondI) {
                    double val2 = instance2.valueSparse(p2);
                    if(attributeRangeMax[idx2] - attributeRangeMin[idx2] != 0) {
                        val2 = (val2 - attributeRangeMin[idx2]) / (attributeRangeMax[idx2] - attributeRangeMin[idx2]);
                    }
                    distanceB += val2 * val2;
                    p2++;

                } else {
                    double val1 = instance1.valueSparse(p1);
                    if(attributeRangeMax[idx1] - attributeRangeMin[idx1] != 0) {
                        val1 = (val1 - attributeRangeMin[idx1]) / (attributeRangeMax[idx1] - attributeRangeMin[idx1]);
                    }
                    distanceA += val1 * val1;
                    p1++;
                }
            }
        }

        if(distanceA != 0.0D && distanceB != 0.0D){
            distance = (double)distance / (Math.sqrt(distanceA) * Math.sqrt(distanceB));
            return 1 - distance;
        }else if(distanceA == 0.0D && distanceB == 0.0D){
            return 0;
        }else {
            return 1;
        }
    }


    /**
     * Cosine similarity of label vectors
     */
    public static double getCosLab(Instance instance1, Instance instance2) {

        double distance = 0.0D;
        double distanceA = 0.0D;
        double distanceB = 0.0D;


        if(instance1.numValues() == instance1.numAttributes()) // Dense Instance
        {
            for(int i = 0; i < instance1.numberOutputTargets(); i++)  // erreur, if instance1.numberOutputTargets() != instance2.numberOutputTargets()
            {
                double val1 = instance1.classValue(i);
                double val2 = instance2.classValue(i);
                distance += val1 * val2;
                distanceA += val1 * val1;
                distanceB += val2 * val2;

            }
        }
        else // Sparse Instance
        {
            int firstI = -1, secondI = -1;
            int firstNumValues  = instance1.numValues();
            int secondNumValues = instance2.numValues();
            int numAttributes   = instance1.numAttributes();
            int numOutputs      = instance1.numOutputAttributes();

            for (int p1 = 0, p2 = 0; p1 < firstNumValues || p2 < secondNumValues;) {

                if (p1 >= firstNumValues) {
                    firstI = numAttributes;
                } else {
                    firstI = instance1.index(p1);
                }

                if (p2 >= secondNumValues) {
                    secondI = numAttributes;
                } else {
                    secondI = instance2.index(p2);
                }

                if (instance1.classIndex() == 0) {
                    if (firstI >= numOutputs && secondI >= numOutputs) {
                        break;
                    }
                }else {
                    if (firstI < instance1.classIndex()) {
                        p1++;
                        continue;
                    }
                    if (secondI < instance1.classIndex()) {
                        p2++;
                        continue;
                    }
                }

                if(firstI == secondI) {

                    double val1 = instance1.valueSparse(p1);
                    double val2 = instance2.valueSparse(p2);

                    distance += val1 * val2;
                    distanceA += val1 * val1;
                    distanceB += val2 * val2;

                    p1++;
                    p2++;
                } else if (firstI > secondI) {
                    double val2 = instance2.valueSparse(p2);
                    distanceB += val2 * val2;
                    p2++;
                } else {
                    double val1 = instance1.valueSparse(p1);
                    distanceA += val1 * val1;
                    p1++;
                }
            }

        }

        if(distanceA != 0.0D && distanceB != 0.0D){
            distance = (double)distance / (Math.sqrt(distanceA) * Math.sqrt(distanceB));
            return 1 - distance;
        }else if(distanceA == 0.0D && distanceB == 0.0D){
            return 0;
        }else {
            return 1;
        }
    }

    /**
     * Distance Euclidienne of feature vectors
     */
    public static double getDistanceAtt(Instance instance1, Instance instance2, double[] attributeRangeMax, double[] attributeRangeMin) {

        double distance = 0.0D;

        if(instance1.numValues() == instance1.numAttributes()) // Dense Instance
        {
            for(int i = 0; i < instance1.numInputAttributes(); i++)
            {
                double val1 = instance1.valueInputAttribute(i);
                double val2 = instance2.valueInputAttribute(i);

                if(attributeRangeMax[i] - attributeRangeMin[i] != 0)
                {
                    val1 = (val1 - attributeRangeMin[i]) / (attributeRangeMax[i] - attributeRangeMin[i]);
                    val2 = (val2 - attributeRangeMin[i]) / (attributeRangeMax[i] - attributeRangeMin[i]);
                    distance += (val1 - val2)*(val1 - val2);
                }
            }
        }else // Sparse Instance
        {

            int firstI = -1, secondI = -1;
            int firstNumValues  = instance1.numValues();
            int secondNumValues = instance2.numValues();
            int numAttributes   = instance1.numAttributes();
            int numOutputs      = instance1.numOutputAttributes();

            for (int p1 = 0, p2 = 0; p1 < firstNumValues || p2 < secondNumValues;) {

                if (p1 >= firstNumValues) {
                    firstI = numAttributes;
                } else {
                    firstI = instance1.index(p1);
                }

                if (p2 >= secondNumValues) {
                    secondI = numAttributes;
                } else {
                    secondI = instance2.index(p2);
                }

                int idx1, idx2;

                if (instance1.classIndex() == 0) {
                    idx1 = firstI - numOutputs;
                    idx2 = secondI - numOutputs;
                }else{
                    idx1 = firstI;
                    idx2 = secondI;
                }

                if (instance1.classIndex() == 0) {
                    if (firstI < numOutputs) {
                        p1++;
                        continue;
                    }
                    if (secondI < numOutputs) {
                        p2++;
                        continue;
                    }
                }else{
                    if (firstI >= instance1.classIndex() && secondI >= instance1.classIndex()) {
                        break;
                    }
                }

                if(firstI == secondI) {

                    double val1 = instance1.valueSparse(p1);
                    double val2 = instance2.valueSparse(p2);
                    if(attributeRangeMax[idx1] - attributeRangeMin[idx1] != 0) {
                        val1 = (val1 - attributeRangeMin[idx1]) / (attributeRangeMax[idx1] - attributeRangeMin[idx1]);
                        val2 = (val2 - attributeRangeMin[idx1]) / (attributeRangeMax[idx1] - attributeRangeMin[idx1]);
                    }

                    distance += (val1 - val2)*(val1 - val2);
                    p1++;
                    p2++;

                } else if (firstI > secondI) {
                    double val2 = instance2.valueSparse(p2);
                    if(attributeRangeMax[idx2] - attributeRangeMin[idx2] != 0) {
                        val2 = (val2 - attributeRangeMin[idx2]) / (attributeRangeMax[idx2] - attributeRangeMin[idx2]);
                    }
                    distance += val2 * val2;
                    p2++;

                } else {
                    double val1 = instance1.valueSparse(p1);
                    if(attributeRangeMax[idx1] - attributeRangeMin[idx1] != 0) {
                        val1 = (val1 - attributeRangeMin[idx1]) / (attributeRangeMax[idx1] - attributeRangeMin[idx1]);
                    }
                    distance += val1 * val1;
                    p1++;

                }
            }
        }

        return Math.sqrt(distance);
    }


    /**
     * Distance Euclidienne of label vectors
     */
    public static double getDistanceLab(Instance instance1, Instance instance2) {

        double distance = 0.0D;

        if(instance1.numValues() == instance1.numAttributes()) // Dense Instance
        {
            for(int i = 0; i < instance1.numberOutputTargets(); i++)  // erreur, if instance1.numberOutputTargets() != instance2.numberOutputTargets()
            {
                double val1 = instance1.classValue(i);
                double val2 = instance2.classValue(i);
                distance += (val1 - val2)*(val1 - val2);

            }
        }
        else // Sparse Instance
        {
            int firstI = -1, secondI = -1;
            int firstNumValues  = instance1.numValues();
            int secondNumValues = instance2.numValues();
            int numAttributes   = instance1.numAttributes();
            int numOutputs      = instance1.numOutputAttributes();

            for (int p1 = 0, p2 = 0; p1 < firstNumValues || p2 < secondNumValues;) {

                if (p1 >= firstNumValues) {
                    firstI = numAttributes;
                } else {
                    firstI = instance1.index(p1);
                }

                if (p2 >= secondNumValues) {
                    secondI = numAttributes;
                } else {
                    secondI = instance2.index(p2);
                }

                if (instance1.classIndex() == 0) {
                    if (firstI >= numOutputs && secondI >= numOutputs) {
                        break;
                    }
                }else {
                    if (firstI < instance1.classIndex()) {
                        p1++;
                        continue;
                    }
                    if (secondI < instance1.classIndex()) {
                        p2++;
                        continue;
                    }
                }

                if(firstI == secondI) {

                    double val1 = instance1.valueSparse(p1);
                    double val2 = instance2.valueSparse(p2);

                    distance += (val1 - val2)*(val1 - val2);

                    p1++;
                    p2++;
                } else if (firstI > secondI) {
                    double val2 = instance2.valueSparse(p2);
                    distance += val2 * val2;
                    p2++;
                } else {
                    double val1 = instance1.valueSparse(p1);
                    distance += val1 * val1;
                    p1++;
                }
            }

        }
        return Math.sqrt(distance);

    }


    public static Instance calculateCenter(Instance centroids, Instance multiLabelInstance, int numMemory, boolean learning) {

        // Rewards (centroids = centroids + lr*(newIns - centroids) = centroids*(1-lr) + lr*newIns
        int rate = 1;
        // punition (centroids = centroids - lr*(newIns - centroids) = centroids*(1+lr) - lr*newIns
        if(!learning) rate = -1;

        if(centroids.numValues() == centroids.numAttributes()) // Dense Instance
        {
            int numAtt = centroids.numAttributes();
            double[] res = new double[numAtt];

            for (int i = 0; i < numAtt; ++i) {

                res[i] = centroids.value(i) * (double) (numMemory-rate);

                res[i] += rate*multiLabelInstance.value(i);

                res[i] /= (double) numMemory;

                if((centroids.classIndex() == 0 && i < centroids.numberOutputTargets()) || (centroids.classIndex() > 0 && i >= centroids.classIndex())) {
                    // Control the label value between 0 and 1
                    if(res[i] < 0){ res[i] = 0.0; }
                    if(res[i] > 1){ res[i] = 1.0; }

                    // Avoid the label value infinitely close to zero
                    if(rate == 1 & res[i]< 0.000001) {res[i] = 0.000001; }
                }
                centroids.setValue(i, res[i]);
            }
            return centroids;
        }
        else // Sparse Instance
        {
            ArrayList<Integer> indexValues = new ArrayList<Integer>();
            ArrayList<Double> attributeValues = new ArrayList<Double>();

            int firstI = -1, secondI = -1;
            int firstNumValues = centroids.numValues();
            int secondNumValues = multiLabelInstance.numValues();
            int numAttributes = centroids.numAttributes();

            for (int p1 = 0, p2 = 0; p1 < firstNumValues || p2 < secondNumValues;) {

                if (p1 >= firstNumValues) {
                    firstI = numAttributes;
                } else {
                    firstI = centroids.index(p1);
                }

                if (p2 >= secondNumValues) {
                    secondI = numAttributes;
                } else {
                    secondI = multiLabelInstance.index(p2);
                }

                if (firstI == secondI) {
                    double centroid = centroids.valueSparse(p1) * (double) (numMemory-rate);
                    centroid += rate*multiLabelInstance.valueSparse(p2);
                    centroid /= (double) numMemory ;
                    attributeValues.add(centroid);
                    indexValues.add(firstI);
                    p1++;
                    p2++;
                } else if (firstI > secondI) {

                    double centroid = rate * multiLabelInstance.valueSparse(p2);
                    centroid /= (double) numMemory ;
                    attributeValues.add(centroid);
                    indexValues.add(secondI);

                    p2++;
                } else {
                    double centroid = centroids.valueSparse(p1) * (double) (numMemory-rate);;
                    centroid /= (double) numMemory;
                    attributeValues.add(centroid);
                    indexValues.add(firstI);
                    p1++;
                }
            }

            int[] index = new int[attributeValues.size()];
            double[] value = new double[attributeValues.size()];

            for (int i = 0; i < attributeValues.size(); i++) {
                index[i] = indexValues.get(i);
                value[i] = attributeValues.get(i);

                if((centroids.classIndex() == 0 && index[i] < centroids.numberOutputTargets()) || (centroids.classIndex() > 0 && index[i] >= centroids.classIndex())) {
                    // Control the label value between 0 and 1
                    if(value[i] < 0){ value[i] = 0.0; }
                    if(value[i] > 1){ value[i] = 1.0; }

                    // Avoid the label value infinitely close to zero
                    if(rate == 1 & value[i]< 0.000001) {value[i] = 0.000001; }
                }
            }
            Instance newkernel = new SparseInstance(1.0, value, index, numAttributes);
            InstancesHeader header = new InstancesHeader(multiLabelInstance.dataset());
            newkernel.setDataset(header);
            return newkernel;
        }
    }

    public static Instance calculateCenter(Instance centroids, Instance multiLabelInstance, int numMemory) {

        int numAtt = centroids.numAttributes();
        double[] res = new double[numAtt];
        for (int i = 0; i < numAtt; ++i) {
            res[i] = centroids.value(i) * (double) (numMemory -1);
            res[i] += multiLabelInstance.value(i);
            res[i] /= (double) numMemory;
            centroids.setValue(i, res[i]);
        }
        return centroids;
    }


    public static Instance calculateCenter(Instance centroids, Instance multiLabelInstance, Instance replaceInstance, int numMemory) {

        int numAtt = centroids.numAttributes();
        double[] res = new double[numAtt];
        for (int i = 0; i < numAtt; ++i) {
            res[i] = centroids.value(i) * (double) numMemory;
            res[i] -= replaceInstance.value(i);
            res[i] += multiLabelInstance.value(i);
            res[i] /= (double) numMemory;
            centroids.setValue(i, res[i]);
        }
        return centroids;
    }

}
