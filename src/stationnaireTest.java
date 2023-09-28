
/**
 * Compare the performance of the models with the test-then-train protocols in the stationary stream.
 * For each data set, the final evaluator information will be shown in the terminal.
 * 'measureCollect' record the evaluation of each prediction to calculate the mean and the standard deviation.
 */

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.MultiLabelPrediction;
import com.yahoo.labs.samoa.instances.Prediction;
import evaluator.MeasureCollect;
import evaluator.PrequentialMultiLabelPerformanceEvaluator;
import moa.classifiers.MultiLabelLearner;
import moa.core.InstanceExample;
import moa.core.TimingUtils;
import moa.streams.MultiTargetArffFileStream;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class PrequentialTest {


    public IntOption sampleFrequencyOption = new IntOption("sampleFrequency", 'f', "How many instances between samples of the learning performance.", 50, 0, 2147483647);
                                                                                                                                                        // 19300; 87856; 13766; 1702; 43907; 13929; 120919; 3782; 2417; 28596
    private int totalInstances;

    public PrequentialTest(int amountInstances) {

        if (amountInstances == -1) {
            this.totalInstances = 2147483647;
        } else {
            this.totalInstances = amountInstances;
        }
    }

    public void run(String data, String position, String algo) throws IOException {
        //load datasets
        MultiTargetArffFileStream stream = new MultiTargetArffFileStream(data, position);
        stream.prepareForUse();

        //prepare algorithm
        MultiLabelLearner learner = BaseLine.getLearner(algo, stream.getHeader());
        learner.setModelContext(stream.getHeader());
        learner.prepareForUse();

        //prepare evaluator
        PrequentialMultiLabelPerformanceEvaluator evaluator = new PrequentialMultiLabelPerformanceEvaluator();
        evaluator.reset();
        evaluator.alphaOption.setValue(1.0);

        // Store the evaluation of each prediction calculate the mean and the standard deviation
        String[] measureName = new String[evaluator.getMeasurements().length];
        for(int i = 0; i < evaluator.getMeasurements().length; i++)
            measureName[i] = evaluator.getMeasurements()[i].getName();
        MeasureCollect measureCollect = new MeasureCollect(measureName);


        long starttime = TimingUtils.getNanoCPUTimeOfCurrentThread();;
        int numberInstances = 0;
        while (numberInstances < totalInstances && stream.hasMoreInstances()) {

            InstanceExample trainInst = (InstanceExample) stream.nextInstance();

            // Estimating models in data streams after 20% (The first 20% of the data is used to optimize the hyperparameters)
            if(numberInstances > this.sampleFrequencyOption.getValue()*20) {

                //Prediction prediction = learner.getPredictionForInstance(trainInst);
                Prediction prediction =  new MultiLabelPrediction(stream.getHeader().numOutputAttributes());
                evaluator.addResult(trainInst, prediction);
                for (int i = 0; i < evaluator.getPerformanceMeasurements().length; i++) {
                    measureCollect.addValue(evaluator.getPerformanceMeasurements()[i].getName(), evaluator.getPerformanceMeasurements()[i].getValue());
                }
                evaluator.reset();
                learner.trainOnInstance(trainInst);
            }

            ++numberInstances;
        }
        long endtime = TimingUtils.getNanoCPUTimeOfCurrentThread();
        double currenttime = TimingUtils.nanoTimeToSeconds((endtime - starttime));
        System.out.println(numberInstances + "\n");
        String timeString = "Times: " + currenttime + " times  \n";

        for(int i = 0; i < evaluator.getPerformanceMeasurements().length; i++)
            System.out.println(evaluator.getPerformanceMeasurements()[i].getName() + "\t" + String.format("%.3f",measureCollect.getMean(i)) + "+" + String.format("%.3f",measureCollect.getError(i)));
        System.out.println(timeString + "\n");

    }

    public static void main(String[] args) throws IOException {
        PrequentialTest batch =  new PrequentialTest(-1); //1000000s

        List<String> algos = Arrays.asList("A2ML");

        //"20NG", "Bibtex", "Bookmarks", "cooking", "Corel16k001", "Enron",  "Eukaryote", "Human", "Imdb", "Mediamill", "Reuters-K500", "Scene", "Slashdot", "TMC2007-500", "Yeast"
        List<String> dataset = Arrays.asList("Bookmarks");
        List<Integer> frequences = Arrays.asList(87856);
        //19300, 7395, 87856, 10491, 13770, 1702, 7766, 3106, 120900, 43910, 6000, 2407, 3782, 28600, 2417


        // original
        //List<String> positions = Arrays.asList( "20", "-159", "-208", "-400", "-153", "-53", "-22", "-14", "28", "-101", "103", "-6", "22", "-22", "-14");
        // random datasets // "20", "159", "208", "400", "153", "53", "22", "14", "28", "101", "103", "6", "22", "22", "14"
        List<String> positions = Arrays.asList( "208");

        for(int i=0; i < algos.size(); i++) {
            String alg = algos.get(i);
            for(int j=0; j < dataset.size(); j++){
                String data = dataset.get(j);
                String position = positions.get(j);
                int fre = frequences.get(j)/100;
                System.out.println("\n" + alg + "\n");
                // String stream = ".././original/" + data + ".arff";
                String stream = ".././stationnary/" + data + ".arff";
                System.out.println(data + "\n");
                batch.sampleFrequencyOption.setValue(fre);
                batch.run(stream, position, alg);
            }
        }
    }
}





