import java.io.*;
import java.util.*;

//Nearest neighbor classifier
public class ClassifyApplicants {
    //set globals to normalize training file values
    final static double MARRIED = 1.0;
    final static double DIVORCED = 0.5;
    final static double SINGLE = 0.0;
    final static double MALE = 1.0;
    final static double FEMALE = 0.0;
    final static int LOW = 1;
    final static int MEDIUM = 2;
    final static int HIGH = 3;
    final static int UNDETERMINED = 4;

    /*************************************************************************/

    //Record class (inner class)
    private class Record {
        private double[] attributes;         //attributes of record      
        private int className;               //class of record

        //Constructor of Record
        private Record(double[] attributes, int className) {
            this.attributes = attributes;    //set attributes 
            this.className = className;      //set class
        }
    }

    /*************************************************************************/

    private int numberRecords;               //number of training records   
    private int numberAttributes;            //number of attributes   
    private int numberClasses;               //number of classes
    private int numberNeighbors;             //number of nearest neighbors
    private ArrayList<Record> records;       //list of training records

    /*************************************************************************/

    //Constructor of ClassifyApplicants
    public ClassifyApplicants() {
        //initial data is empty           
        numberRecords = 0;
        numberAttributes = 0;
        numberClasses = 0;
        numberNeighbors = 0;
        records = null;
    }

    /*************************************************************************/

    void loadTrainingDataExcept(String trainingFile, int index) throws FileNotFoundException {
        Scanner inFile = new Scanner(new File(trainingFile));

        //read number of records, attributes, classes
        numberRecords = inFile.nextInt();
        numberAttributes = inFile.nextInt();
        numberClasses = inFile.nextInt();

        //create empty list of records
        records = new ArrayList<Record>();

        //for each record
        for (int i = 0; i < numberRecords; i++) {
            if (i == index) {
                continue;
            }
            //create attribute array
            double[] attributeArray = new double[numberAttributes];

            //read attribute values
            for (int j = 0; j < numberAttributes; j++)
                attributeArray[j] = inFile.nextDouble();

            //read class name
            int className = inFile.nextInt();

            //create record
            Record record = new Record(attributeArray, className);

            //add record to list of records
            records.add(record);
        }

        inFile.close();
    }

    //Method loads data from training file
    public void loadTrainingData(String trainingFile) throws IOException {
        Scanner inFile = new Scanner(new File(trainingFile));

        //read number of records, attributes, classes
        numberRecords = inFile.nextInt();
        numberAttributes = inFile.nextInt();
        numberClasses = inFile.nextInt();

        //create empty list of records
        records = new ArrayList<Record>();

        //for each record
        for (int i = 0; i < numberRecords; i++) {
            //create attribute array
            double[] attributeArray = new double[numberAttributes];

            //read attribute values
            for (int j = 0; j < numberAttributes; j++)
                attributeArray[j] = inFile.nextDouble();

            //read class name
            int className = inFile.nextInt();

            //create record
            Record record = new Record(attributeArray, className);

            //add record to list of records
            records.add(record);
        }

        inFile.close();
    }

    /*************************************************************************/

    //Method sets number of nearest neighbors
    public void setParameters(int numberNeighbors) {
        this.numberNeighbors = numberNeighbors;
    }

    /*************************************************************************/

    //Method reads records from test file, determines their classes, 
    //and writes classes to classified file
    public void classifyData(String testFile, String classifiedFile) throws IOException {
        Scanner inFile = new Scanner(new File(testFile));
        PrintWriter outFile = new PrintWriter(new FileWriter(classifiedFile));

        //read number of records
        int numberRecords = inFile.nextInt();

        //write number of records
        outFile.println(numberRecords);

        //for each record
        for (int i = 0; i < numberRecords; i++) {
            //create attribute array
            double[] attributeArray = new double[numberAttributes];

            //read attribute values
            for (int j = 0; j < numberAttributes; j++)
                attributeArray[j] = inFile.nextDouble();

            //find class of attributes
            int className = classify(attributeArray);

            //write class name
            outFile.println(convertIntToClass(className));
        }

        inFile.close();
        outFile.close();
    }

    /*************************************************************************/

    //Method determines the class of a set of attributes
    private int classify(double[] attributes) {
        double[] distance = new double[numberRecords];
        int[] id = new int[numberRecords];

        //find distances between attributes and all records
        for (int i = 0; i < numberRecords - 1; i++) {
            distance[i] = distance(attributes, records.get(i).attributes);
            id[i] = i;
        }

        //find nearest neighbors
        nearestNeighbor(distance, id);

        //find majority class of nearest neighbors
        int className = majority(id);

        //return class
        return className;
    }

    /*************************************************************************/

    //Method finds the nearest neighbors
    private void nearestNeighbor(double[] distance, int[] id) {
        //sort distances and choose nearest neighbors
        for (int i = 0; i < numberNeighbors; i++)
            for (int j = i; j < numberRecords; j++)
                if (distance[i] > distance[j]) {
                    double tempDistance = distance[i];
                    distance[i] = distance[j];
                    distance[j] = tempDistance;

                    int tempId = id[i];
                    id[i] = id[j];
                    id[j] = tempId;
                }
    }

    /*************************************************************************/

    //Method finds the majority class of nearest neighbors
    private int majority(int[] id) {
        double[] frequency = new double[numberClasses];

        //class frequencies are zero initially
        for (int i = 0; i < numberClasses; i++)
            frequency[i] = 0;

        //each neighbor contributes 1 to its class
        for (int i = 0; i < numberNeighbors; i++)
            frequency[records.get(id[i]).className - 1] += 1;

        //find majority class
        int maxIndex = 0;
        for (int i = 0; i < numberClasses; i++)
            if (frequency[i] > frequency[maxIndex]) maxIndex = i;

        return maxIndex + 1;
    }

    /*************************************************************************/

    //Method finds Euclidean distance between two points
    private double distance(double[] u, double[] v) {
        double distance = 0;

        for (int i = 0; i < u.length; i++)
            distance = distance + (u[i] - v[i]) * (u[i] - v[i]);

        distance = Math.sqrt(distance);

        return distance;
    }

    /*************************************************************************/

    //Method validates classifier using validation file and displays error rate
    public double validate(String validationFile) throws IOException {
        Scanner inFile = new Scanner(new File(validationFile));

        //read number of records
        int numberRecords = inFile.nextInt();
        inFile.nextLine();
        //initially zero errors
        int numberErrors = 0;

        //for each record
        for (int i = 0; i < numberRecords; i++) {
            // Update training data to be all records except `i`.
            loadTrainingDataExcept(validationFile, i);

            double[] attributeArray = new double[numberAttributes];

            //read attributes
            for (int j = 0; j < numberAttributes; j++)
                attributeArray[j] = inFile.nextDouble();

            //read actual class
            int actualClass = inFile.nextInt();

            //find class predicted by classifier
            int predictedClass = classify(attributeArray);

            //error if predicted and actual classes do not match
            if (predictedClass != actualClass) numberErrors += 1;
        }

        //find and print error rate
        double errorRate = 100.0 * numberErrors / numberRecords;

        System.out.println("validation error: " + errorRate + "%");

        inFile.close();
        return errorRate;
    }

    /************************************************************************/

    public static void main(String[] args) throws IOException {
        Scanner userInput = new Scanner(System.in);
        String normalizedTrainingFile = "output/trainingFile.txt";
        String normalizedTestFile = "output/testFile.txt";

        System.out.print("Enter training file: ");
        String trainingFile = userInput.nextLine();
        System.out.print("Enter test file: ");
        String testFile = userInput.nextLine();
        System.out.println("Enter output file: ");
        String outputFile = "output/" + userInput.nextLine();

        int kValue = 7;

        convertTrainingFile(trainingFile, normalizedTrainingFile);
        convertTestFile(testFile, normalizedTestFile);

        ClassifyApplicants classifier = new ClassifyApplicants();

        //load training data
        classifier.loadTrainingData(normalizedTrainingFile);

        //set nearest neighbors
        classifier.setParameters(kValue);

        //validate classifier
        double validationError = classifier.validate(normalizedTrainingFile);

        //classify test data
        classifier.classifyData(normalizedTestFile, outputFile);

        PrintWriter outFile = new PrintWriter(new FileWriter(outputFile, true));
        outFile.printf("\nValidation Error:  %.2f%%\n", validationError);
        outFile.println("K-Value: " + kValue);
        outFile.close();
        userInput.close();
    }

    //Method converts training file to numerical format
    private static void convertTrainingFile(String inputFile, String outputFile) throws IOException {
        //input and output files
        Scanner inFile = new Scanner(new File(inputFile));
        PrintWriter outFile = new PrintWriter(new FileWriter(outputFile));

        //read number of records, attributes, classes
        int numberRecords = inFile.nextInt();
        int numberAttributes = inFile.nextInt();
        int numberClasses = inFile.nextInt();

        //write number of records, attributes, classes
        outFile.println(numberRecords + " " + numberAttributes + " " + numberClasses);

        //for each record
        for (int i = 0; i < numberRecords; i++) {
            int score = inFile.nextInt();                      //convert score
            double scoreNumber = convertCreditScore(score);
            outFile.print(scoreNumber + " ");

            int income = inFile.nextInt();                  //convert income
            double incomeNumber = convertIncome(income);
            outFile.print(incomeNumber + " ");

            int age = inFile.nextInt();
            double ageNumber = convertAge(age);             //convert age
            outFile.print(ageNumber + " ");

            String sex = inFile.next();                  //convert sex
            double sexNumber = convertSex(sex);
            outFile.print(sexNumber + " ");

            String status = inFile.next();
            double statusNumber = convertStatus(status);
            outFile.print(statusNumber + " ");

            String className = inFile.next();                  //convert class name
            int classNumber = convertClassToInt(className);
            outFile.println(classNumber);
        }

        inFile.close();
        outFile.close();
    }

    //Method converts test file to numerical format
    private static void convertTestFile(String inputFile, String outputFile) throws IOException {
        //input and output files
        Scanner inFile = new Scanner(new File(inputFile));
        PrintWriter outFile = new PrintWriter(new FileWriter(outputFile));

        //read number of records
        int numberRecords = inFile.nextInt();

        //write number of records
        outFile.println(numberRecords);

        //for each record
        for (int i = 0; i < numberRecords; i++) {
            int score = inFile.nextInt();                      //convert score
            double scoreNumber = convertCreditScore(score);
            outFile.print(scoreNumber + " ");

            int income = inFile.nextInt();                  //convert income
            double incomeNumber = convertIncome(income);
            outFile.print(incomeNumber + " ");

            int age = inFile.nextInt();
            double ageNumber = convertAge(age);             //convert age
            outFile.print(ageNumber + " ");

            String sex = inFile.next();                  //convert sex
            double sexNumber = convertSex(sex);
            outFile.print(sexNumber + " ");

            String status = inFile.next();
            double statusNumber = convertStatus(status);
            outFile.println(statusNumber + " ");
        }
            inFile.close();
            outFile.close();

    }

    //converts attributes to be in the range of 0-1
    private static double convertCreditScore(int score) {
        return (double) (score - 500) / 400;
    }

    private static double convertIncome(int income) {
        return (double) (income - 30) / 60;
    }

    private static double convertAge(int age) {
        return (double) (age - 30) / 50;
    }

    //convert rest of attributes to match that
    private static double convertSex(String sex) {
        if (sex.equals("male")) {
            return MALE;
        }
        return FEMALE;
    }

    private static double convertStatus(String status){
        return switch (status) {
            case "married" -> MARRIED;
            case "divorced" -> DIVORCED;
            default -> SINGLE;
        };
    }

    private static int convertClassToInt(String className){
        return switch (className) {
            case "low" -> LOW;
            case "medium" -> MEDIUM;
            case "high" -> HIGH;
            default -> UNDETERMINED;
        };
    }
    private static String convertIntToClass(int classValue){
        return switch (classValue) {
            case LOW -> "low";
            case  MEDIUM -> "medium";
            case HIGH -> "high";
            default -> "undetermined";
        };
    }
}

