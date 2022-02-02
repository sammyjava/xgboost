package org.ncgr.xgboost;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

/**
 * Run XGBoost on a pair (train/test) of LIBSVM formatted files.
 */
public class XGBoostSVM {

    /**
     * Main class for running all static methods.
     */
    public static void main(String[] args) throws Exception {
	Options options = new Options();
        CommandLineParser parser = new DefaultParser();
        HelpFormatter formatter = new HelpFormatter();
        CommandLine cmd;

	// files
        Option trainFileOption = new Option("train", "trainfile", true, "input LIBSVM format file with training data (required)");
        trainFileOption.setRequired(true);
        options.addOption(trainFileOption);
        // 
	Option testFileOption = new Option("test", "testfile", true, "input LIBSVM format file with testing data (required)");
	testFileOption.setRequired(false);
	options.addOption(testFileOption);
        // parameters
        Option nroundOption = new Option("nround", "nround", true, "number of rounds (4)");
        nroundOption.setRequired(false);
        options.addOption(nroundOption);
        //
        Option etaOption = new Option("eta", "eta", true, "eta parameter (0.5)");
        etaOption.setRequired(false);
        options.addOption(etaOption);
        //
        Option maxDepthOption = new Option("max_depth", "max_depth", true, "max_depth parameter (6)");
        maxDepthOption.setRequired(false);
        options.addOption(maxDepthOption);
        //
        Option objectiveOption = new Option("objective", "objective", true, "objective parameter (binary:logitraw)");
        objectiveOption.setRequired(false);
        options.addOption(objectiveOption);
        //
        Option gammaOption = new Option("gamma", "gamma", true, "gamma parameter (0.5)");
        gammaOption.setRequired(false);
        options.addOption(gammaOption);
        //
        Option subSampleOption = new Option("subsample", "subsample", true, "subsample parameter (0.6)");
        subSampleOption.setRequired(false);
        options.addOption(subSampleOption);
        //
        Option samplingMethodOption = new Option("sampling_method", "sampling_method", true, "sampling_method parameter (uniform)");
        samplingMethodOption.setRequired(false);
        options.addOption(samplingMethodOption);
        //
        Option treeMethodOption = new Option("tree_method", "tree_method", true, "tree_method parameter (auto)");
        treeMethodOption.setRequired(false);
        options.addOption(treeMethodOption);
        //
        Option verbosityOption = new Option("verbosity", "verbosity", true, "verbosity parameter (0)");
        verbosityOption.setRequired(false);
        options.addOption(verbosityOption);
        //
        Option minChildWeightOption = new Option("min_child_weight", "min_child_weight", true, "min_child_weight parameter (0.9)");
        minChildWeightOption.setRequired(false);
        options.addOption(minChildWeightOption);
        
        try {
            cmd = parser.parse(options, args);
        } catch (ParseException e) {
            System.err.println(e.getMessage());
            formatter.printHelp("XGBoostSVM", options);
            System.exit(1);
            return;
        }

        // files
        String trainFilename = cmd.getOptionValue("trainfile");
        String testFilename = cmd.getOptionValue("testfile");

        // parameters
        int nround = 4;
        if (cmd.hasOption("nround")) nround = Integer.parseInt(cmd.getOptionValue("nround"));

        Map<String,Object> params = new HashMap<>();
        params.put("eval_metric", "error"); // binary classification
        if (cmd.hasOption("verbosity")) {
            params.put("verbosity", Integer.parseInt(cmd.getOptionValue("verbosity")));
        } else {
            params.put("verbosity", 0);
        }
        if (cmd.hasOption("eta")) {
            params.put("eta", Double.parseDouble(cmd.getOptionValue("eta")));
        } else {
            params.put("eta", 0.5);
        }
        if (cmd.hasOption("max_depth")) {
            params.put("max_depth", Integer.parseInt(cmd.getOptionValue("max_depth")));
        } else {
            params.put("max_depth", 6);
        }
        if (cmd.hasOption("objective")) {
            params.put("objective", cmd.getOptionValue("objective"));
        } else {
            params.put("objective", "binary:logitraw");
        }
        if (cmd.hasOption("gamma")) {
            params.put("gamma", Double.parseDouble(cmd.getOptionValue("gamma")));
        } else {
            params.put("gamma", 0.5);
        }
        if (cmd.hasOption("sampling_method")) {
            params.put("sampling_method", cmd.getOptionValue("sampling_method"));
        } else {
            params.put("sampling_method", "uniform");
        }
        if (cmd.hasOption("tree_method")) {
            params.put("tree_method", cmd.getOptionValue("tree_method"));
        } else {
            params.put("tree_method", "auto");
        }
        if (cmd.hasOption("subsample")) {
            params.put("subsample", Double.parseDouble(cmd.getOptionValue("subsample")));
        } else {
            params.put("subsample", 0.6);
        }
        if (cmd.hasOption("min_child_weight")) {
            params.put("min_child_weight", Double.parseDouble(cmd.getOptionValue("min_child_weight")));
        } else {
            params.put("min_child_weight", 0.9);
        }
        // input matrices
        DMatrix trainMat = new DMatrix(trainFilename);
        DMatrix testMat = new DMatrix(testFilename);

        // Specify a watch list to see model accuracy on data sets
        Map<String, DMatrix> watches = new HashMap<String, DMatrix>() {
                {
                    put("train", trainMat);
                    put("test", testMat);
                }
            };

        // // 10-fold cross-validatation
        // int nfold = 10;
        // String[] metrics = null;
        // String[] xvResults = XGBoost.crossValidation(trainMat, params, nround, nfold, metrics, null, null);
        // List<String> resultsList = Arrays.asList(xvResults);
        // System.out.println(resultsList);

        // train
        Booster booster = XGBoost.train(trainMat, params, nround, watches, null, null);

        // predict
        float[][] predicts = booster.predict(testMat);

        // get case/control status of each record
        List<Boolean> statusList = Util.readSVMStatus(testFilename);
        int numCase = 0;
        int numCtrl = 0;
        int numTP = 0;
        int numFP = 0;
        int numTN = 0;
        int numFN = 0;
        for (int i=0; i<statusList.size(); i++) {
            String label = "";
            String result = "";
            if (statusList.get(i)) {
                numCase++;
                label = "case";
                if (predicts[i][0]>0.500) {
                    result = "TP";
                    numTP++;
                } else {
                    result = "FN";
                    numFN++;
                }
            } else {
                numCtrl++;
                label = "ctrl";
                if (predicts[i][0]<0.500) {
                    result = "TN";
                    numTN++;
                } else {
                    result = "FP";
                    numFP++;
                }
            }
            // System.err.println(label+"\t"+result+"\t"+predicts[i][0]);
        }

        System.err.println(params);
        
        // output
        Util.printResults(numCase, numCtrl, numTP, numTN, numFP, numFN);
    }
}
