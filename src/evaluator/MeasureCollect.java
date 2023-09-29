package evaluator;

import moa.AbstractMOAObject;
import moa.cluster.Clustering;
import moa.evaluation.MembershipMatrix;
import moa.gui.visualization.DataPoint;

import java.util.ArrayList;
import java.util.HashMap;


import java.util.ArrayList;
import java.util.HashMap;
import moa.AbstractMOAObject;

import static java.lang.Double.NaN;

public class MeasureCollect extends AbstractMOAObject {
    private String[] names;
    private ArrayList<Double>[] values;
    private double[] sumValues;
    private HashMap<String, Integer> map;
    private int numMeasures = 0;

    public MeasureCollect(String[] names) {
        this.names = names;
        this.numMeasures = this.names.length;
        this.map = new HashMap(this.numMeasures);

        int i;
        for(i = 0; i < this.names.length; ++i) {
            this.map.put(this.names[i], i);
        }

        this.values = new ArrayList[this.numMeasures];
        this.sumValues = new double[this.numMeasures];

        for(i = 0; i < this.numMeasures; ++i) {
            this.values[i] = new ArrayList();
            this.sumValues[i] = 0.0D;
        }
    }

    public void addValue(int index, double value) {

        if (Double.isNaN(value)){ value = 0.0D; }
        this.values[index].add(value);
        double[] var10000 = this.sumValues;
        var10000[index] += value;
    }

    public void addValue(String name, double value) {
        if (this.map.containsKey(name)) {
            this.addValue((Integer)this.map.get(name), value);
        } else {
            System.out.println(name + " is not a valid measure key, no value added");
        }
    }

    public int getNumMeasures() {
        return this.numMeasures;
    }

    public String getName(int index) {
        return this.names[index];
    }

    public double getLastValue(int index) {
        return this.values[index].size() < 1 ? 0.0D / 0.0 : (Double)this.values[index].get(this.values[index].size() - 1);
    }

    public double getMean(int index) {
        return this.values[index].size() >= 1 ? this.sumValues[index] / (double)this.values[index].size() : 0.0D / 0.0;
    }

    public double getEcartType(int index) {
        double mean = getMean(index);
        double sumVariance = 0.0D;

        for(int i = 0; i < this.values[index].size(); ++i) {
            double v = (Double)this.values[index].get(i);
            double d = (v - mean)*(v - mean);
            sumVariance += d;
        }
        return this.values[index].size() >= 1 ? Math.sqrt(sumVariance / (double)this.values[index].size()) : 0.0D / 0.0;
    }

    public double getError(int index){
        double ecartType = this.getEcartType(index);
        return 2*ecartType/Math.sqrt((double)this.values[index].size() -1);
    }


    public int getNumberOfValues(int index) {
        return this.values[index].size();
    }

    public double getValue(int index, int i) {
        return i >= this.values[index].size() ? 0.0D / 0.0 : (Double)this.values[index].get(i);
    }

    public ArrayList<Double> getAllValues(int index) {
        return this.values[index];
    }

    public void getDescription(StringBuilder sb, int indent) {
    }

}

