package evaluator;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.MultiLabelInstance;
import com.yahoo.labs.samoa.instances.Prediction;
import moa.AbstractMOAObject;
import moa.core.Example;
import moa.core.Measurement;
import moa.core.Utils;
import moa.evaluation.MultiTargetPerformanceEvaluator;

public class WindowMultiLabelPerformanceEvaluator extends AbstractMOAObject implements MultiTargetPerformanceEvaluator {

    protected int L;

    protected int posWindow;
    protected int lenWindow;
    protected int SizeWindow;
    protected double qtyNaNs;

    /** running sum of accuracy */
    protected double[] windowExactMatch;
    private double sumExactMatch;
    private double sumHamming;
    protected double[] windowHamming;
    private double[] sumTP;
    private double[] sumFP;
    private double[] sumFN;

    private double sumExamplePrecision, sumExampleRecall, sumExampleAccuracy;
    protected double[] windowExamplePrecision, windowExampleRecall, windowExampleAccuracy;
    private double microPrecision, microRecall, microFScore;
    private double macroPrecision, macroRecall, macroFScore;
    protected double[][] windowSumTP, windowSumFP, windowSumFN;

    public WindowMultiLabelPerformanceEvaluator(int windowSize){
        this.SizeWindow = windowSize;
    }

	
	protected double b;

    public void reset() {
        sumTP = new double[L];
        sumFP = new double[L];
        sumFN = new double[L];
        sumExactMatch = 0;
        sumHamming = 0;
        sumExampleAccuracy = 0;
        sumExamplePrecision = 0;
        sumExampleRecall = 0;

        this.windowExactMatch = new double[this.SizeWindow];
        this.windowHamming = new double[this.SizeWindow];
        this.windowExamplePrecision = new double[this.SizeWindow];
        this.windowExampleRecall = new double[this.SizeWindow];
        this.windowExampleAccuracy = new double[this.SizeWindow];
        this.windowSumTP = new double[L][this.SizeWindow];
        this.windowSumFP = new double[L][this.SizeWindow];
        this.windowSumFN = new double[L][this.SizeWindow];

        this.posWindow = 0;
        this.lenWindow = 0;

    }

    public void addResult(Example<Instance> example, Prediction y) {

        MultiLabelInstance x = (MultiLabelInstance) example.getData();

        if (L == 0) {
            L = x.numberOutputTargets();
            reset();
        }

        if (y == null) {
            System.err.print("[WARNING] Prediction is null! (Ignoring this prediction)");
        }
        else if (y.numOutputAttributes() < x.numOutputAttributes()) {
            System.err.println("[WARNING] Only "+y.numOutputAttributes()+" labels found! (Expecting "+x.numOutputAttributes()+")\n (Ignoring this prediction)");
        }
        else {
            int correct = 0;
            double cur_tp = 0;
            double cur_fp = 0;
            double cur_fn = 0;

            for (int j = 0; j < y.numOutputAttributes(); j++) {
                int yp = (y.getVote(j,1) >= 0.5) ? 1 : 0;  //prediction itself
                int y_true = (int) x.classValue(j); // actual label

                //True Positive if 1 and predicted 1:
                cur_tp   += (y_true == 1 && yp == 1) ? 1 : 0;   //for example-based
                sumTP[j] = add(cur_tp, sumTP[j], windowSumTP[j]);
                windowSumTP[j][this.posWindow] = cur_tp;


                //False Negative if 1 and predicted 0:
                cur_fn   += (y_true == 1 && yp == 0) ? 1 : 0;   //for example-based
                sumFN[j] = add(cur_fn, sumFN[j], windowSumFN[j]);
                windowSumFN[j][this.posWindow] = cur_fn;


                //False Positive if 0 and predicted 1:
                cur_fp   += (y_true == 0 && yp == 1) ? 1 : 0;   //for example-based
                sumFP[j] = add(cur_fp, sumFP[j], windowSumFP[j]);
                windowSumFP[j][this.posWindow] = cur_fp;

                correct  += (y_true == yp) ? 1 : 0;
            }

            double delta_ham = correct / (double)L;
            double delta_EM = (correct == L) ? 1 : 0;

            sumHamming  = add(delta_ham, sumHamming, windowHamming);
            windowHamming[this.posWindow] = delta_ham;

            sumExactMatch  = add(delta_EM, sumExactMatch, windowExactMatch); 		// Exact Match
            windowExactMatch[this.posWindow] = delta_EM;

            double delta_exPre = 0.0, delta_exRec = 0.0, delta_exAcc = 0.0;
            
        	if(cur_tp + cur_fp > 0)
        	{
                delta_exPre = cur_tp / (cur_tp + cur_fp);
                sumExamplePrecision = add(delta_exPre, sumExamplePrecision, windowExamplePrecision);
                windowExamplePrecision[this.posWindow] = delta_exPre;
        	}

        	if(cur_tp + cur_fn > 0)
        	{
                delta_exRec = cur_tp / (cur_tp + cur_fn);
                sumExampleRecall = add(delta_exRec, sumExampleRecall, windowExampleRecall);
                windowExampleRecall[this.posWindow] = delta_exRec;
        	}

        	if(cur_tp + cur_fn + cur_fp > 0)
        	{
                delta_exAcc = cur_tp / (cur_tp + cur_fn + cur_fp);
                sumExampleAccuracy  = add(delta_exAcc, sumExampleAccuracy, windowExampleAccuracy);
                windowExampleAccuracy[this.posWindow] = delta_exAcc;
            }

            ++this.posWindow;
            if (this.posWindow == this.SizeWindow) {
                this.posWindow = 0;
            }

            if (this.lenWindow < this.SizeWindow) {
                ++this.lenWindow;
            }
        }
    }


    public double add(double value, double sum, double[] window) {
        double forget = window[this.posWindow];
        if (!Double.isNaN(forget)) {
            sum -= forget;
        }
        if (!Double.isNaN(value)) {
            sum += value;
        }
        return sum;
    }



    private double getMicroPrecision(double[] tp, double[] fp){
        double sum_tp = 0;
        double sum_tp_fp = 0.0;

        for (int i = 0; i < L; i++) {
            sum_tp += tp[i];
            sum_tp_fp += tp[i] + fp[i];
        }

        double micro_precision = sum_tp_fp > 0 ? sum_tp / sum_tp_fp : 0;
        return micro_precision;
    }

    private double getMicroRecall(double[] tp, double[] fn){
        double sum_tp = 0;
        double sum_tp_fn = 0.0;
        double micro_recall = 0.0;

        for (int i = 0; i < L; i++) {
            sum_tp += tp[i];
            sum_tp_fn += tp[i] + fn[i];
        }

        micro_recall = sum_tp_fn > 0 ? sum_tp / sum_tp_fn : 0;
        return micro_recall;
    }

    private double getMacroPrecision(double[] tp, double[] fp){
        double[] macro_precision = tp.clone();
        for (int i = 0; i < L; i++) {
            double denom = tp[i] + fp[i];

            if(denom == 0.0)
                macro_precision[i] = 0.0;
            else
                macro_precision[i] = tp[i] / (tp[i] + fp[i]);
        }

        return Utils.sum(macro_precision) / L;
    }

    private double getMacroRecall(double[] tp, double[] fn){
        double[] macro_recall = tp.clone();
        for (int i = 0; i < L; i++) {
            double denom = tp[i] + fn[i];

            if(denom == 0.0)
                macro_recall[i] = 0.0;
            else
                macro_recall[i] = tp[i] / (tp[i] + fn[i]);
        }

        return Utils.sum(macro_recall) / L;
    }

    public Measurement[] getPerformanceMeasurements() {
        double examplePrecision = sumExamplePrecision / this.lenWindow;
        double exampleRecall = sumExampleRecall / this.lenWindow;
        double exampleFScore = (examplePrecision + exampleRecall) > 0 ? 2.0 * examplePrecision * exampleRecall / (examplePrecision + exampleRecall) : 0;

        // micro averaged measures:
        microPrecision = getMicroPrecision(sumTP, sumFP);   //micro averaged precision
        microRecall = getMicroRecall(sumTP, sumFN);         //micro averaged recall

        microFScore = 0;                                    //micro averaged fscore
        if(microPrecision + microRecall != 0.0)
            microFScore = 2 * microPrecision * microRecall / (microPrecision + microRecall);

        // macro averaged measures:
        macroPrecision = getMacroPrecision(sumTP, sumFP);   //macro averaged precision
        macroRecall = getMacroRecall(sumTP, sumFN);         //macro averaged recall

        macroFScore = 0;                                    //macro averaged fscore
        if(macroPrecision + macroRecall != 0.0)
            macroFScore = 2 * macroPrecision * macroRecall / (macroPrecision + macroRecall);

        // Measurements
        Measurement m[] = new Measurement[]{
                new Measurement("Subset Accuracy", sumExactMatch / this.lenWindow),
                new Measurement("Hamming Score", sumHamming / this.lenWindow),
                new Measurement("Example-Based Accuracy", sumExampleAccuracy / this.lenWindow),
                new Measurement("Example-Based Precision", examplePrecision),
                new Measurement("Example-Based Recall", exampleRecall),
                new Measurement("Example-Based F-Measure", exampleFScore),
                new Measurement("Micro-Averaged Precision", microPrecision),
                new Measurement("Micro-Averaged Recall", microRecall),
                new Measurement("Micro-Averaged F-Measure", microFScore),
                new Measurement("Macro-Averaged Precision", macroPrecision),
                new Measurement("Macro-Averaged Recall", macroRecall),
                new Measurement("Macro-Averaged F-Measure", macroFScore)
        };

        return m;
    }

    public void getDescription(StringBuilder sb, int indent) {
        sb.append("Multi-label Performance Evaluator");
    }

    public void addResult(Example<Instance> example, double[] classVotes) {
        // NOTHING
    }
}