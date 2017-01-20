/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.RandomForest;

import weka.core.Instances;

/**
 *
 * @author samy
 */
public class Main{

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        BufferedReader br = null;
        int numFolds = 10;
        br = new BufferedReader(new FileReader("src/stories-for-weka.arff"));

        Instances trainData = new Instances(br);
        Instances tempTrainData;

//        System.out.println(trainData.numAttributes());

        int attributeCount = trainData.numAttributes();

        trainData.setClassIndex(attributeCount - 1);
        br.close();
        RandomForest rf = new RandomForest();
//        rf.set
//        rf.setNumTrees(100);

        //   rf.buildClassifier(trainData);

        for (int i=0; i < attributeCount - 1; i++){

            tempTrainData = trainData;
//            Instances att = tempTrainData.(i);

            Evaluation evaluation = new Evaluation(tempTrainData);
            evaluation.crossValidateModel(rf, tempTrainData, numFolds, new Random(1));


            System.out.println(evaluation.toSummaryString("\nResults when removing att. No " + i +" named " + trainData.attribute(i) +"\n======\n", true));
            System.out.println(evaluation.toClassDetailsString());
            System.out.println("Results For Class -1- ");
            System.out.println("Precision=  " + evaluation.precision(0));
            System.out.println("Recall=  " + evaluation.recall(0));
            System.out.println("F-measure=  " + evaluation.fMeasure(0));
            System.out.println("Results For Class -2- ");
            System.out.println("Precision=  " + evaluation.precision(1));
            System.out.println("Recall=  " + evaluation.recall(1));
            System.out.println("F-measure=  " + evaluation.fMeasure(1));
        }


    }


//    @Override
//    public Object clone() throws CloneNotSupportedException{
//        return super.clone();
//    }
}