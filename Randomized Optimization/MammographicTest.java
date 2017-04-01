package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer
 * or more than 15 rings.
 *
 * @author Hannah Lau
 * @version 1.0
 */
public class MammographicTest {
    private static int numTests = 15;

    private static Instance[] instances = initializeInstances();

    private static int inputLayer = 5, hiddenLayer = 2, outputLayer = 1, trainingIterations = 10000;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[15];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[15];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[15];
    private static String[] oaNames = { "RHC"
                                        , "SA_temp_1E6", "SA_temp_1E8","SA_temp_1E10","SA_temp_1E12"
                                        , "SA_rate_70","SA_rate_80","SA_rate_90","SA_rate_95"
                                        , "GA_population_100", "GA_population_200", "GA_population_300"
                                        , "GA_mate_50", "GA_mate_75", "GA_mate_100"};
    private static String results = "";
    private static String txtResults = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    private static int trials = 5;
    private static int iter = 20000;
    private static int iterGA = 1000;
    private static double rate = 1.15;

    private static String[][] RHC_Acc = new String[trials][iter];
    private static String[][] RHC_Train = new String[trials][iter];
    private static String[][] RHC_Test = new String[trials][iter];
    private static String[][][] SA_Acc = new String[8][trials][iter];
    private static String[][][] SA_Train = new String[8][trials][iter];
    private static String[][][] SA_Test = new String[8][trials][iter];
    private static String[][][] GA_Acc = new String[6][trials][iterGA];
    private static String[][][] GA_Train = new String[6][trials][iterGA];
    private static String[][][] GA_Test = new String[6][trials][iterGA];


    public static void main(String[] args) {
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E7, .90, nnop[1]);
        oa[2] = new SimulatedAnnealing(1E9, .90, nnop[2]);
        oa[3] = new SimulatedAnnealing(1E11, .90, nnop[3]);
        oa[4] = new SimulatedAnnealing(1E13, .90, nnop[4]);
        oa[5] = new SimulatedAnnealing(1E11, .70, nnop[5]);
        oa[6] = new SimulatedAnnealing(1E11, .80, nnop[6]);
        oa[7] = new SimulatedAnnealing(1E11, .90, nnop[7]);
        oa[8] = new SimulatedAnnealing(1E11, .95, nnop[8]);
        oa[9] = new StandardGeneticAlgorithm(100, 100, 10, nnop[9]);
        oa[10] = new StandardGeneticAlgorithm(200, 100, 10, nnop[10]);
        oa[11] = new StandardGeneticAlgorithm(300, 100, 10, nnop[11]);
        oa[12] = new StandardGeneticAlgorithm(200, 80, 10, nnop[12]);
        oa[13] = new StandardGeneticAlgorithm(200, 100, 10, nnop[13]);
        oa[14] = new StandardGeneticAlgorithm(200, 120, 10, nnop[14]);

        //RHC
        System.out.println("RHC");
        for (int i = 0; i < 1; i++) {
            for (int z = 0; z < trials; z++) {
                for (int k = 1; k < iter; k = (int)(rate * k) + 10) {
                    trainingIterations = k;
                    System.out.println("TrainingIterations: " + trainingIterations);
                    double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
                    train(oa[i], networks[i], oaNames[i]); //trainer.train();
                    end = System.nanoTime();
                    trainingTime = end - start;
                    trainingTime /= Math.pow(10,9);

                    Instance optimalInstance = oa[i].getOptimal();
                    networks[i].setWeights(optimalInstance.getData());

                    double predicted, actual;
                    start = System.nanoTime();
                    for(int j = 0; j < instances.length; j++) {
                        networks[i].setInputValues(instances[j].getData());
                        networks[i].run();

                        predicted = Double.parseDouble(instances[j].getLabel().toString());
                        actual = Double.parseDouble(networks[i].getOutputValues().toString());

                        double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

                    }
                    end = System.nanoTime();
                    testingTime = end - start;
                    testingTime /= Math.pow(10,9);

                    RHC_Acc[z][k] = df.format(correct/(correct+incorrect)*100);
                    RHC_Train[z][k] = df.format(trainingTime);
                    RHC_Test[z][k] = df.format(testingTime);
                }
            }
        }
        try {
            PrintWriter accwriter = new PrintWriter("../data/RHC/MammographicResults_RHC_acc.txt", "UTF-8");
            PrintWriter trainwriter = new PrintWriter("../data/RHC/MammographicResults_RHC_train.txt", "UTF-8");
            PrintWriter testwriter = new PrintWriter("../data/RHC/MammographicResults_RHC_test.txt", "UTF-8");
            accwriter.println(oaNames[0]);
            trainwriter.println(oaNames[0]);
            testwriter.println(oaNames[0]);
            for (int i = 0; i < iter; i++) {
                if (RHC_Acc[0][i] != null) {
                    accwriter.print(i);
                    trainwriter.print(i);
                    testwriter.print(i);
                    for (int f = 0; f < trials; f++) {
                        accwriter.print("\t" + RHC_Acc[f][i]);
                        trainwriter.print("\t" + RHC_Train[f][i]);
                        testwriter.print("\t" + RHC_Test[f][i]);
                    }
                    accwriter.println();
                    trainwriter.println();
                    testwriter.println();
                }
            }
            accwriter.close();
            trainwriter.close();
            testwriter.close();
        } catch (IOException e) {}
        //SA
        System.out.println("SA");
        for (int i = 1; i < oa.length - 6; i++) {
            int temp = i - 1;
            for (int z = 0; z < trials; z++) {
                 for (int k = 1; k < iter; k = (int)(rate * k) + 10) {
                    trainingIterations = k;
                    System.out.println("TrainingIterations: " + trainingIterations);
                    double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
                    train(oa[i], networks[i], oaNames[i]); //trainer.train();
                    end = System.nanoTime();
                    trainingTime = end - start;
                    trainingTime /= Math.pow(10,9);

                    Instance optimalInstance = oa[i].getOptimal();
                    networks[i].setWeights(optimalInstance.getData());

                    double predicted, actual;
                    start = System.nanoTime();
                    for(int j = 0; j < instances.length; j++) {
                        networks[i].setInputValues(instances[j].getData());
                        networks[i].run();

                        predicted = Double.parseDouble(instances[j].getLabel().toString());
                        actual = Double.parseDouble(networks[i].getOutputValues().toString());

                        double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

                    }
                    end = System.nanoTime();
                    testingTime = end - start;
                    testingTime /= Math.pow(10,9);

                    SA_Acc[temp][z][k] = df.format(correct/(correct+incorrect)*100);
                    SA_Train[temp][z][k] = df.format(trainingTime);
                    SA_Test[temp][z][k] = df.format(testingTime);
                }
            }
        }
        try {
            PrintWriter accwriter = new PrintWriter("../data/SA/MammographicResults_SA_acc.txt", "UTF-8");
            PrintWriter trainwriter = new PrintWriter("../data/SA/MammographicResults_SA_train.txt", "UTF-8");
            PrintWriter testwriter = new PrintWriter("../data/SA/MammographicResults_SA_test.txt", "UTF-8");
            for (int p = 1; p < oa.length - 6; p++) {
                accwriter.println(oaNames[p]);
                trainwriter.println(oaNames[p]);
                testwriter.println(oaNames[p]);
                int temp = p - 1;
                for (int i = 0; i < iter; i++) {
                    if (SA_Acc[temp][0][i] != null) {
                        accwriter.print(i);
                        trainwriter.print(i);
                        testwriter.print(i);
                        for (int f = 0; f < trials; f++) {
                            accwriter.print("\t" + SA_Acc[temp][f][i]);
                            trainwriter.print("\t" + SA_Train[temp][f][i]);
                            testwriter.print("\t" + SA_Test[temp][f][i]);
                        }
                        accwriter.println();
                        trainwriter.println();
                        testwriter.println();
                    }
                }
            }
            accwriter.close();
            trainwriter.close();
            testwriter.close();
        } catch (IOException e) {}
        //GA
        System.out.println("GA");
        for (int i = oa.length - 6; i < oa.length; i++) {
            int temp = i - 9;
            for (int z = 0; z < trials; z++) {
                for (int k = 1; k < iterGA; k = (int)(rate * k) + 10) {
                    trainingIterations = k;
                    System.out.println("TrainingIterations: " + trainingIterations);
                    double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
                    train(oa[i], networks[i], oaNames[i]); //trainer.train();
                    end = System.nanoTime();
                    trainingTime = end - start;
                    trainingTime /= Math.pow(10,9);

                    Instance optimalInstance = oa[i].getOptimal();
                    networks[i].setWeights(optimalInstance.getData());

                    double predicted, actual;
                    start = System.nanoTime();
                    for(int j = 0; j < instances.length; j++) {
                        networks[i].setInputValues(instances[j].getData());
                        networks[i].run();

                        predicted = Double.parseDouble(instances[j].getLabel().toString());
                        actual = Double.parseDouble(networks[i].getOutputValues().toString());

                        double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

                    }
                    end = System.nanoTime();
                    testingTime = end - start;
                    testingTime /= Math.pow(10,9);

                    GA_Acc[temp][z][k] = df.format(correct/(correct+incorrect)*100);
                    GA_Train[temp][z][k] = df.format(trainingTime);
                    GA_Test[temp][z][k] = df.format(testingTime);
                }
            }
        }
        try {
            PrintWriter accwriter = new PrintWriter("../data/GA/MammographicResults_GA_acc.txt", "UTF-8");
            PrintWriter trainwriter = new PrintWriter("../data/GA/MammographicResults_GA_train.txt", "UTF-8");
            PrintWriter testwriter = new PrintWriter("../data/GA/MammographicResults_GA_test.txt", "UTF-8");
            for (int p = oa.length - 6; p < oa.length; p++) {
                accwriter.println(oaNames[p]);
                trainwriter.println(oaNames[p]);
                testwriter.println(oaNames[p]);
                int temp = p - 9;
                for (int i = 0; i < iterGA; i++) {
                    if (GA_Acc[temp][0][i] != null) {
                        accwriter.print(i);
                        trainwriter.print(i);
                        testwriter.print(i);
                        for (int f = 0; f < trials; f++) {
                            accwriter.print("\t" + GA_Acc[temp][f][i]);
                            trainwriter.print("\t" + GA_Train[temp][f][i]);
                            testwriter.print("\t" + GA_Test[temp][f][i]);
                        }
                        accwriter.println();
                        trainwriter.println();
                        testwriter.println();
                    }
                }
            }
            accwriter.close();
            trainwriter.close();
            testwriter.close();
        } catch (IOException e) {}
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        //System.out.println("\nError results for " + oaName + "\n---------------------------");

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double error = 0;
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }

            //System.out.println(df.format(error));
        }
    }

    private static Instance[] initializeInstances() {

        double[][][] attributes = new double[806][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("../data/Mammographic.txt")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[5]; // 5 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < 5; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            // classifications range from 0 to 30; split into 0 - 14 and 15 - 30
            instances[i].setLabel(new Instance(attributes[i][1][0]));
        }

        return instances;
    }
}
