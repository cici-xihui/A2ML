import A2ML.A2ML;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import evaluator.PrequentialMultiLabelPerformanceEvaluator;
import meka.classifiers.multilabel.incremental.PSUpdateable;
import meka.classifiers.multilabel.incremental.CCUpdateable;
import moa.classifiers.MultiLabelLearner;
import moa.classifiers.multilabel.MEKAClassifier;
import moa.classifiers.multilabel.MajorityLabelset;
import moa.classifiers.multilabel.MultilabelHoeffdingTree;
import moa.classifiers.multilabel.meta.OzaBagAdwinML;
import moa.classifiers.multilabel.meta.OzaBagML;
import moa.classifiers.multilabel.trees.ISOUPTree;
import moa.classifiers.rules.multilabel.AMRulesMultiLabelClassifier;
import moa.classifiers.rules.multilabel.meta.MultiLabelRandomAMRules;
import moa.core.Measurement;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import moa.core.*;


/**
 *
 * @author Joel
 */
public class BaseLine {

    public static MultiLabelLearner getLearner(String method, InstancesHeader header){
        if(method.equals("majori")){
            MajorityLabelset learner = new MajorityLabelset();
            learner.setModelContext(header);
            learner.prepareForUse();
            learner.resetLearningImpl();
            return learner;

        }else if(method.equals("PS")){
            MEKAClassifier learner = new MEKAClassifier();
            PSUpdateable baseLearner = new PSUpdateable();
            learner.baseLearnerOption.setCurrentObject(baseLearner);
            learner.setModelContext(header);
            learner.prepareForUse();
            learner.resetLearningImpl();
            return learner;

        }else if(method.equals("CC")){
            MEKAClassifier learner = new MEKAClassifier();
            CCUpdateable baseLearner = new CCUpdateable();
            learner.baseLearnerOption.setCurrentObject(baseLearner);
            learner.setModelContext(header);
            learner.prepareForUse();
            learner.resetLearningImpl();
            return learner;

        }else if(method.equals("EaBR")){
            OzaBagAdwinML learner = new OzaBagAdwinML();
            MEKAClassifier baseMeka = new MEKAClassifier();
            baseMeka.setModelContext(header);
            baseMeka.prepareForUse();
            baseMeka.resetLearningImpl();
            learner.baseLearnerOption.setCurrentObject(baseMeka);
            learner.setModelContext(header);
            learner.prepareForUse();
            learner.resetLearningImpl();
            return learner;

        }else if(method.equals("MLHT")){
            MultilabelHoeffdingTree learner = new MultilabelHoeffdingTree();
            MEKAClassifier PS = new MEKAClassifier();
            MajorityLabelset majori = new MajorityLabelset();
            majori.setModelContext(header);
            majori.prepareForUse();
            majori.resetLearningImpl();
            learner.learnerOption.setCurrentObject(PS);
            learner.setModelContext(header);
            learner.prepareForUse();
            learner.resetLearningImpl();
            return learner;

        }else if(method.equals("HTps")){
            MultilabelHoeffdingTree learner = new MultilabelHoeffdingTree();
            MEKAClassifier PS = new MEKAClassifier();
            PSUpdateable psUpdateable = new PSUpdateable();
            PS.baseLearnerOption.setCurrentObject(psUpdateable);
            PS.setModelContext(header);
            PS.prepareForUse();
            PS.resetLearningImpl();
            learner.learnerOption.setCurrentObject(PS);
            learner.setModelContext(header);
            learner.prepareForUse();
            learner.resetLearningImpl();
            return learner;

        }else if(method.equals("EaHTps")){
            OzaBagAdwinML learner = new OzaBagAdwinML();
            MultilabelHoeffdingTree MLHT = new MultilabelHoeffdingTree();
            MEKAClassifier PS = new MEKAClassifier();
            PSUpdateable psUpdateable = new PSUpdateable();
            PS.baseLearnerOption.setCurrentObject(psUpdateable);
            PS.setModelContext(header);
            PS.prepareForUse();
            PS.resetLearningImpl();
            MLHT.learnerOption.setCurrentObject(PS);
            MLHT.setModelContext(header);
            MLHT.prepareForUse();
            MLHT.resetLearningImpl();
            learner.baseLearnerOption.setCurrentObject(MLHT);
            learner.setModelContext(header);
            learner.prepareForUse();
            learner.resetLearningImpl();
            return learner;

        }else if(method.equals("BagHTps")){
            OzaBagML learner = new OzaBagML();
            MultilabelHoeffdingTree MLHT = new MultilabelHoeffdingTree();
            MEKAClassifier PS = new MEKAClassifier();
            PSUpdateable psUpdateable = new PSUpdateable();
            PS.baseLearnerOption.setCurrentObject(psUpdateable);
            PS.setModelContext(header);
            PS.prepareForUse();
            PS.resetLearningImpl();
            MLHT.learnerOption.setCurrentObject(PS);
            MLHT.setModelContext(header);
            MLHT.prepareForUse();
            MLHT.resetLearningImpl();
            learner.baseLearnerOption.setCurrentObject(MLHT);
            learner.setModelContext(header);
            learner.prepareForUse();
            learner.resetLearningImpl();
            return learner;

        }else if(method.equals("EaCC")){
            OzaBagAdwinML learner = new OzaBagAdwinML();
            MEKAClassifier baseMeka = new MEKAClassifier();
            CCUpdateable baseLearner = new CCUpdateable();
            baseMeka.baseLearnerOption.setCurrentObject(baseLearner);
            baseMeka.setModelContext(header);
            baseMeka.prepareForUse();
            baseMeka.resetLearningImpl();
            learner.baseLearnerOption.setCurrentObject(baseMeka);
            learner.setModelContext(header);
            learner.prepareForUse();
            learner.resetLearningImpl();
            return learner;

        }else if(method.equals("EaPS")){
            OzaBagAdwinML learner = new OzaBagAdwinML();
            MEKAClassifier baseMeka = new MEKAClassifier();
            PSUpdateable baseLearner = new PSUpdateable();
            baseMeka.baseLearnerOption.setCurrentObject(baseLearner);
            baseMeka.setModelContext(header);
            baseMeka.prepareForUse();
            baseMeka.resetLearningImpl();
            learner.baseLearnerOption.setCurrentObject(baseMeka);
            learner.setModelContext(header);
            learner.prepareForUse();
            learner.resetLearningImpl();
            return learner;

        }else if(method.equals("ISOUPTree")){
            ISOUPTree learner = new ISOUPTree();
            learner.setModelContext(header);
            learner.prepareForUse();
            learner.resetLearningImpl();
            return learner;

        }else if(method.equals("EaISOUPTree")){
            OzaBagAdwinML learner = new OzaBagAdwinML();
            ISOUPTree base = new ISOUPTree();
            base.setModelContext(header);
            base.prepareForUse();
            base.resetLearningImpl();
            learner.baseLearnerOption.setCurrentObject(base);
            learner.setModelContext(header);
            learner.prepareForUse();
            learner.resetLearningImpl();
            return learner;

        } else if(method.equals("AMRules")){
            AMRulesMultiLabelClassifier learner = new AMRulesMultiLabelClassifier();
            learner.setModelContext(header);
            learner.prepareForUse();
            learner.resetLearningImpl();
            return learner;

        }else if(method.equals("RandomAMRules")){
            MultiLabelRandomAMRules learner = new MultiLabelRandomAMRules();
            AMRulesMultiLabelClassifier base = new AMRulesMultiLabelClassifier();
            base.setModelContext(header);
            base.prepareForUse();
            base.resetLearningImpl();
            learner.baseLearnerOption.setCurrentObject(base);
            learner.setModelContext(header);
            learner.prepareForUse();
            learner.resetLearningImpl();
            return learner;

        } else if(method.equals("A2ML")){
            A2ML learner = new A2ML();
            learner.setModelContext(header);
            learner.prepareForUse();
            learner.resetLearningImpl();
            return learner;
        }


        return null;

    }

    public static void executeMethod(String method, String dataset, MultiLabelLearner learner, long starttime, PrequentialMultiLabelPerformanceEvaluator evaluator) throws IOException{
        String outdir = "./out/results/" + dataset + "-alg-" + method + "-";
//        File directory = new File(dir);

        String outFileName = outdir + "-results" + ".txt";
        BufferedWriter writer = new BufferedWriter(new FileWriter(new File(outFileName)));
        StringBuilder out;


        long endtime = TimingUtils.getNanoCPUTimeOfCurrentThread();
        out = new StringBuilder();

        System.out.println(learner.getPurposeString());
        System.out.println("Performance Measurements:");

        Measurement[] measurements = evaluator.getPerformanceMeasurements();
        Measurement.getMeasurementsDescription(measurements, out, 0);
        System.out.println(out.toString() + "\n");

        writer.write(learner.getPurposeString() + "\n");
        writer.write(String.valueOf(out));
        writer.write("\n");

        String timeString = "Time: " + TimingUtils.nanoTimeToSeconds((endtime - starttime)) + " s  \n";
        System.out.println(timeString + "\n");
        writer.write(timeString + "\n");


        writer.write("Size of the model: " + SizeOf.fullSizeOf(learner) + " bytes\n");

        writer.close();
    }


}

