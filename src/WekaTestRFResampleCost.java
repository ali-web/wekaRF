import weka.classifiers.CostMatrix;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.CostSensitiveClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.Randomize;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;

/**
 *
 * @author samy
 */
public class WekaTestRFResampleCost {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        BufferedReader br = null;
        double numFolds = 10.0d;
        double precisionOne = 0.0d;
        double recallOne = 0.0d;
        double precisionTwo = 0.0d;
        double recallTwo = 0.0d;

        double recallTwoRounded, precisionTwoRounded, WA;


        String project = "JF";
        String stage = "3rd";
        String WI_Type = "plan";
        String attribute = "none";
        int regularAttributeCount = 27;

        //String stageStr = new WekaTestRFResampleCost().getXthNumber(stageNum);

        int[] CVSeedValues = {3, 11, 20, 22, 36, 51, 64, 71, 77, 95,
                               7, 15, 21, 39, 44, 52, 68, 79, 88, 92,
                              106, 110, 124, 133, 149, 154, 160, 175, 182, 196
                                };


        String url = "jdbc:mysql://127.0.0.1:3306/CLM_h2";
        String user = "root";
        String password = "";

        // Load the Connector/J driver
        Class.forName("com.mysql.jdbc.Driver").newInstance();
        // Establish connection to MySQL
        Connection conn = DriverManager.getConnection(url, user, password);

        String query = "INSERT INTO vir_loo (project, wi_type, attribute, stage, seed, recall, `precision`, performance) VALUES (?, ?, ?, ?, ?, ?, ?, ?)";

        Instances       inst;
        Instances       instNew;
        //Remove          remove;

        //loop over attributes and exclude them in turn
        for (int a = 0; a < regularAttributeCount; a++) {

            //loop over seed values for CV
            for (int c = 0; c < CVSeedValues.length; c++) {
                precisionTwo = 0.0d;
                recallTwo = 0.0d;


                br = new BufferedReader(new FileReader("src/" + WI_Type + "-" + project + "-" + stage + ".arff"));
                ArffSaver saverTets = new ArffSaver();
                ArffSaver saverTraining = new ArffSaver();
                Instances trainData = new Instances(br);
                trainData.setClassIndex(trainData.numAttributes() - 1);
                br.close();

                if (a == 0) {
                    attribute = "none";
                } else {

                    attribute = trainData.attribute(a).name();
                    //code to remove an attribute
                    String[] options = new String[2];
                    options[0] = "-R";                                    // "range"
                    options[1] = Integer.toString(a);                                     // first attribute
                    Remove remove = new Remove();                         // new instance of filter
                    remove.setOptions(options);                           // set options
                    remove.setInputFormat(trainData);                          // inform filter about dataset **AFTER** setting options
                    trainData = Filter.useFilter(trainData, remove);   // apply filter
                }



                Randomize randFilterMain = new Randomize();
                randFilterMain.setRandomSeed(CVSeedValues[c]);

                randFilterMain.setInputFormat(trainData);
                trainData = Filter.useFilter(trainData, randFilterMain);
                int size = (int) (trainData.numInstances() / numFolds);
                int begin = 0; // is index if flod.
                int end = size - 1; // is index

                System.out.println("Total Size of instances" + trainData.numInstances() + " , flod size=" + size);
                for (int i = 1; i <= numFolds; i++) {
                    System.out.println("Iteration # " + i + " Begin =" + begin + " , end=" + end);
                    Instances tempTraining = new Instances(trainData);
                    Instances tempTesting = new Instances(trainData, begin, (end - begin));
                    for (int j = 0; j < (end - begin); j++) {
                        tempTraining.delete(begin);
                    }

                    //// Filters
                    Resample resample = new Resample();

                    resample.setBiasToUniformClass(0.5f);
                    resample.setInvertSelection(false);
                    resample.setNoReplacement(false);

                    resample.setRandomSeed(1);
                    //smoteFilter.setClassValue("2");
                    resample.setInputFormat(tempTraining);

                    System.out.println("Number of instances before filter " + tempTraining.numInstances());

                    Instances resmapleTempTraining = Filter.useFilter(tempTraining, resample);


                    System.out.println("Number of instances after filter " + resmapleTempTraining.numInstances());

                    RandomForest randomForest = new RandomForest();
                    //            randomForest.setNumTrees(100);
                    randomForest.setNumIterations(50);

                    System.out.println("Started building the model #" + i);
                    //            randomForest.buildClassifier(resmapleTempTraining);

                    CostSensitiveClassifier costSensitiveClassifier = new CostSensitiveClassifier();
                    CostMatrix costMatrix = new CostMatrix(2);
                    //          costMatrix.setCell(0, 0, 0.8d);
                    //          costMatrix.setCell(0, 1, 5.0d);
                    costMatrix.setCell(0, 1, 3d);
                    //            costMatrix.setCell(1, 1, 1.0d);

                    costSensitiveClassifier.setClassifier(randomForest);
                    costSensitiveClassifier.setCostMatrix(costMatrix);
                    costSensitiveClassifier.buildClassifier(resmapleTempTraining);

                    saverTraining.setInstances(resmapleTempTraining);
                    saverTraining.setFile(new File("src" + i + "_training.arff"));
                    saverTets.setInstances(tempTesting);
                    saverTets.setFile(new File("src" + i + "_testing.arff"));

                    saverTraining.writeBatch();
                    saverTets.writeBatch();


                    System.out.println("Done with building the model");

                    Evaluation evaluation = new Evaluation(tempTesting);

                    evaluation.evaluateModel(costSensitiveClassifier, tempTesting);

                    System.out.println("Results For Class -1- ");
                    System.out.println("Precision=  " + evaluation.precision(0));
                    System.out.println("Recall=  " + evaluation.recall(0));
                    System.out.println("Results For Class -2- ");
                    System.out.println("Precision=  " + evaluation.precision(1));
                    System.out.println("Recall=  " + evaluation.recall(1));
                    precisionOne += evaluation.precision(0);
                    recallOne += evaluation.recall(0);
                    precisionTwo += evaluation.precision(1);
                    recallTwo += evaluation.recall(1);


                    begin = end + 1;
                    end += size;
                    if (i == (numFolds - 1)) {
                        end = trainData.numInstances();
                    }
                }

                recallTwoRounded = round(recallTwo / numFolds, 3);
                precisionTwoRounded = round(precisionTwo / numFolds, 3);

                System.out.println("####################################################");
                System.out.println("Results For Class -1- YES ");
                System.out.println("Precision=  " + precisionOne / numFolds);
                System.out.println("Recall=  " + recallOne / numFolds);


                System.out.println("Results For Class -2- NO ");
                System.out.println("Precision=  " + precisionTwo / numFolds);
                System.out.println("Recall=  " + recallTwo / numFolds);

                WA = round((2 * precisionTwoRounded + recallTwoRounded) / 3, 3);

                PreparedStatement ps = conn.prepareStatement(query);
                ps.setString(1, project);
                ps.setString(2, WI_Type);
                ps.setString(3, attribute); //attribute
                ps.setString(4, stage);
                ps.setInt(5, CVSeedValues[c]);
                ps.setDouble(6, recallTwoRounded);
                ps.setDouble(7, precisionTwoRounded);
                ps.setDouble(8, WA);
                ps.executeUpdate();

            }
        }
    }


    private String getXthNumber(int num){
        switch (num){
            case 0: return "0th";
            case 1: return "1st";
            case 2: return "2nd";
            case 3: return "3rd";
        }

        return null;
    }

    public static double round(double value, int places) {
        if (places < 0) throw new IllegalArgumentException();

        BigDecimal bd = new BigDecimal(value);
        bd = bd.setScale(places, RoundingMode.HALF_UP);
        return bd.doubleValue();
    }
}