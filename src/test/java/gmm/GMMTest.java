package gmm;

import gmm.FGMM;
import gmm.FGMM.ComponentParams;

import org.jblas.DoubleMatrix;
import org.junit.Test;

//under construction
public class GMMTest {
    FGMM gmm;

    @Test
    public void driver() {
        int nComponents = 5;
        int dim = 2;
        // initialize mu and sigma
        DoubleMatrix[] mu = new DoubleMatrix[nComponents];
        DoubleMatrix[] sigma = new DoubleMatrix[nComponents];
        mu = new DoubleMatrix[] { new DoubleMatrix(new double[] { 0, 0 }), new DoubleMatrix(new double[] { 10, 0 }),
                new DoubleMatrix(new double[] { 2, 0 }), new DoubleMatrix(new double[] { 0, 0 }),
                new DoubleMatrix(new double[] { -1, 6 }) };
        for (int i = 0; i < nComponents; i++) {
            sigma[i] = new DoubleMatrix(new double[][] { { 1, 0 }, { 0, 1 } });
        }

        // generate data
        int sampleSize = 1000;
        DoubleMatrix[] sample = new DoubleMatrix[sampleSize];
        int[] label = new int[sampleSize];
        for (int i = 0; i < sampleSize; i++) {
            if (i % 2 == 0) {
                sample[i] = DoubleMatrix.randn(dim);
                label[i] = 0;
            } else if (i % 3 == 0) {
                sample[i] = DoubleMatrix.randn(dim).add(20);
                label[i] = 1;
            } else if (i % 5 == 0) {
                sample[i] = DoubleMatrix.randn(dim).add(-5);
                label[i] = 2;
            } else if (i % 7 == 0) {
                sample[i] = DoubleMatrix.randn(dim).add(30);
                label[i] = 3;
            } else {
                sample[i] = DoubleMatrix.randn(dim).add(-10);
                label[i] = 3;
            }
        }

        DoubleMatrix muZero = DoubleMatrix.zeros(dim);
        double kappa = 0.5;
        DoubleMatrix sigmaZero = DoubleMatrix.eye(dim);
        int nu = 4;
        double alpha = 0.5;
        int nIter = 100;

        gmm = new FGMM(dim, nComponents, muZero, kappa, sigmaZero, nu, alpha);
        ComponentParams param = gmm.learn(sample, mu, sigma, nIter);
        System.out.println("mu1=" + param.mu[0]);
        System.out.println("sigma1=" + param.sigma[0]);
        System.out.println("mu2=" + param.mu[1]);
        System.out.println("sigma2=" + param.sigma[1]);
        System.out.println("mu3=" + param.mu[2]);
        System.out.println("sigma3=" + param.sigma[2]);
        System.out.println("mu4=" + param.mu[3]);
        System.out.println("sigma4=" + param.sigma[3]);
        System.out.println("mu5=" + param.mu[4]);
        System.out.println("sigma5=" + param.sigma[4]);

    }

}
