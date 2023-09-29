/**
 * Compare the performance of the models with the test-then-train protocols in the non-stationary stream.
 * For each data set, the final evaluator information will be shown in the terminal.
 * 'measureCollect' record the evaluation of each prediction to calculate the mean and the standard deviation.
 */


import com.github.javacliparser.FileOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Prediction;
import evaluator.PrequentialMultiLabelPerformanceEvaluator;
import moa.classifiers.MultiLabelLearner;
import moa.core.InstanceExample;
import moa.core.Measurement;
import moa.core.SizeOf;
import moa.core.TimingUtils;
import moa.evaluation.LearningEvaluation;
import moa.evaluation.preview.LearningCurve;
import moa.streams.MultiTargetArffFileStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.List;

public class driftTest {

    public FileOption dumpFileOption = new FileOption("dumpFile", 'd', "File to append intermediate csv results to.", (String)null, "csv", true);
    public IntOption sampleFrequencyOption = new IntOption("sampleFrequency", 'f', "How many instances between samples of the learning performance.", 50, 0, 2147483647);
                                                                                                                                                        // 19300; 87856; 13766; 1702; 43907; 13929; 120919; 3782; 2417; 28596

    private int totalInstances;

    public driftTest(int amountInstances) {

        if (amountInstances == -1) {
            this.totalInstances = 2147483647;
        } else {
            this.totalInstances = amountInstances;
        }
    }

        public void run(String data, String position, String algo) throws IOException {

        File dumpFile = this.dumpFileOption.getFile();
        PrintStream immediateResultStream = null;
        boolean firstDump = true;
        if (dumpFile != null) {
            try {
                if (dumpFile.exists()) {
                    immediateResultStream = new PrintStream(new FileOutputStream(dumpFile, true), true);
                } else {
                    immediateResultStream = new PrintStream(new FileOutputStream(dumpFile), true);
                }
            } catch (Exception var36) {
                throw new RuntimeException("Unable to open immediate result file: " + dumpFile, var36);
            }
        }
        LearningCurve learningCurve = new LearningCurve("learning evaluation instances");
        MultiTargetArffFileStream stream = new MultiTargetArffFileStream(data, position);
        stream.prepareForUse();

        stream.prepareForUse();

        MultiLabelLearner learner = BaseLine.getLearner(algo, stream.getHeader());

        learner.setModelContext(stream.getHeader());
        learner.prepareForUse();

        PrequentialMultiLabelPerformanceEvaluator evaluator = new PrequentialMultiLabelPerformanceEvaluator();
        evaluator.reset();
        evaluator.alphaOption.setValue(1.0);

        long starttime = TimingUtils.getNanoCPUTimeOfCurrentThread();
        int numberInstances = 0;
        while(numberInstances < totalInstances && stream.hasMoreInstances()) {


            InstanceExample trainInst = (InstanceExample) stream.nextInstance();
            Prediction prediction = learner.getPredictionForInstance(trainInst);
            evaluator.addResult(trainInst, prediction);

            if (numberInstances % (long)this.sampleFrequencyOption.getValue() == 0L && numberInstances != 0 ) {
                learningCurve.insertEntry(new LearningEvaluation(new Measurement[]{new Measurement("learning evaluation instances", (double)numberInstances)}, evaluator, learner));
                if (immediateResultStream != null) {
                    if (firstDump) {
                        immediateResultStream.println(learningCurve.headerToString());
                        firstDump = false;
                    }
                    immediateResultStream.println(learningCurve.entryToString(learningCurve.numEntries() - 1));
                    immediateResultStream.flush();
                }

                evaluator.reset();
            }

            learner.trainOnInstance(trainInst);
            ++numberInstances;

        }

        learningCurve.insertEntry(new LearningEvaluation(new Measurement[]{new Measurement("learning evaluation instances", (double)numberInstances)}, evaluator, learner));
        if (immediateResultStream != null) {
            if (firstDump) {
                immediateResultStream.println(learningCurve.headerToString());
                firstDump = false;
            }

            immediateResultStream.println(learningCurve.entryToString(learningCurve.numEntries() - 1));
            immediateResultStream.flush();
        }

        if (immediateResultStream != null) {
            immediateResultStream.close();
        }


        System.out.println(numberInstances + " instances.");
        long endtime = TimingUtils.getNanoCPUTimeOfCurrentThread();

        for(int i = 0; i < evaluator.getPerformanceMeasurements().length; i++)
            System.out.println(evaluator.getPerformanceMeasurements()[i].getName() + "\t" + String.format("%.3f",evaluator.getPerformanceMeasurements()[i].getValue()));

        String timeString = "Time: " + TimingUtils.nanoTimeToSeconds((endtime - starttime)) + " s  \n";
        System.out.println(timeString + "\n");
        System.out.println("Size of the model: " + SizeOf.fullSizeOf(learner) + " bytes\n");
    }

    public static void main(String[] args) throws IOException {

        driftTest batch =  new driftTest(-1); //1000000s
        List<String> algos = Arrays.asList("HTps", "A2ML", "ODM", "EaHTps"); //"HTps","ISOUPTree", "AMRules", "MLSAMPkNN", "MLSAkNN", "A2ML", "ODM", "EaHTps", "EaISOUPTree", "GOOWEML", "AESAKNNS"
        List<String> dataset = Arrays.asList("Bookmarks"); //  "Booamarks", "Imdb"
        List<String> positions = Arrays.asList("208"); //  "208", "28"
        List<Integer> frequences = Arrays.asList(130000); // (120000, 130000)
        for(int i=0; i < algos.size(); i++){
            String alg = algos.get(i);
            System.out.println("\n" + alg + "\n");
            for(int j=0; j < dataset.size(); j++) {
                String data = dataset.get(j);
                String position = positions.get(j);
                int fre = frequences.get(j)/100;
                String stream = ".././nonStationnary/article/" + "xGraduel.arff";
                String adresse = "./out/drift/xGraduel/" + alg + ".csv";
                batch.sampleFrequencyOption.setValue(fre);
                batch.dumpFileOption.setValue(adresse);
                batch.run(stream, position, alg);
            }
        }
    }
}





