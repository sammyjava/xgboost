package org.ncgr.xgboost;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileNotFoundException;
import java.io.IOException;

import java.text.DecimalFormat;

import java.util.List;
import java.util.ArrayList;

/**
 * Static utility methods.
 */
public class Util {
    // number formats
    static final DecimalFormat sci = new DecimalFormat("0.00E00");
    static final DecimalFormat perc = new DecimalFormat("0.0%");
    static final DecimalFormat rate = new DecimalFormat("0.000");
    static final DecimalFormat round = new DecimalFormat("0");
    static final String pm = "\u00B1";

    /**
     * Print prediction results.
     */
    public static void printResults(double correct, double accuracy, double TPR, double FPR, double MCC, double precision) {
    	System.out.println("correct\taccur\tTPR\tFPR\tMCC\tprec");
	System.out.println(round.format(correct)+"\t"+perc.format(accuracy)+"\t"+rate.format(TPR)+"\t"+rate.format(FPR)+"\t"+rate.format(MCC)+"\t"+rate.format(precision));
    }

    /**
     * Print prediction results.
     */
    public static void printResults(int cases, int controls, int tp, int tn, int fp, int fn) {
        int correct = tp + tn;
        double accuracy = (double)correct / (cases+controls);
        double tpr = (double)tp / cases;
        double fpr = (double)fp / controls;
        double mcc = Util.mcc(tp, tn, fp, fn);
        double precision = (double)tp / (tp+fp);
        printResults(correct, accuracy, tpr, fpr, mcc, precision);
    }

    /**
     * Read a List of Boolean case/control=true/false status from a LIBSVM format file
     *
     * 0 0:0 1:1 2:0 3:2 ...
     * 0 0:1 1:1 2:1 3:0 ...
     * 1 0:0 1:2 2:0 3:0 ...
     * 1 0:2 1:0 2:0 3:1 ...
     */
    public static List<Boolean> readSVMStatus(String filename) throws FileNotFoundException, IOException {
        List<Boolean> statusList = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        String line = null;
        while ((line=reader.readLine())!=null) {
            String[] fields = line.split(" ");
            int stat = Integer.parseInt(fields[0]);
            boolean status = false;
            if (stat==0) {
                status = false; // ctrl
            } else if (stat==1) {
                status = true;  // case
            } else {
                System.err.println("ERROR: case/control status found="+stat+" in "+filename+".");
                System.exit(1);
            }
            statusList.add(status);
        }
        return statusList;
    }

    /**
     * Compute the MCC.
     */
    public static double mcc(int tp, int tn, int fp, int fn) {
        return (double)(tp*tn-fp*fn) / Math.sqrt((double)(tp+fp)*(double)(tp+fn)*(double)(tn+fp)*(double)(tn+fn));
    }
}
