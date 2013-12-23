package gmm;

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.jblas.DoubleMatrix;

public class Util {

    /**
     * Sampling from Wishart 
     * @param psi
     * @param nu
     * @return (nu,psi.columns) matrix
     */
    public static DoubleMatrix sampleWishart(DoubleMatrix psi, int nu) {
    	
        //sample X(i,:) ~ N(0,psi) for 0<=i<nu then X^T * X ~ Wishart(psi,nu)
        int dimention = psi.columns;
        DoubleMatrix zeros = DoubleMatrix.zeros(dimention);

        MultivariateNormalDistribution normal = new MultivariateNormalDistribution(zeros.toArray(), psi.toArray2());
        
        DoubleMatrix sample = new DoubleMatrix(nu, dimention);
        for (int i = 0; i < nu; i++) {
            sample.putRow(i, new DoubleMatrix(normal.sample()));
        }

        return sample.transpose().mmul(sample);
    }

    /**
     * Sampling from Categorical
     * 
     * @param prob
     * @return i, 0<=i<=prob.length-1
     */
    public static int sampleCategorical(double[] prob) {

        // normalize probability
        double sumProb = 0;
        for (int i = 0; i < prob.length; i++) {
            sumProb += prob[i];
        }
        for (int i = 0; i < prob.length; i++) {
            prob[i] = prob[i] / sumProb;
        }

        // set cumulative probability
        double[] cumulativeProb = new double[prob.length];
        cumulativeProb[0] = prob[0];
        for (int i = 1; i < prob.length; i++) {
            cumulativeProb[i] = prob[i] + cumulativeProb[i - 1];
        }

        // generate proposal between 0 and 1
        MersenneTwister random = new MersenneTwister();
        double proposal = random.nextDouble();

        int i = 0;
        while (true) {
            if (proposal < cumulativeProb[i]) {
                // accept proposal
                break;
            }
            i++;
        }
        return i;
    }
}
