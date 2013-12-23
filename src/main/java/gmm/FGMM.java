package gmm;

import gmm.Util;

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Finite Gaussian Mixture Models. (with Gibbs sampling)
 * 
 * @author khaito
 */
public class FGMM {
    private static final Logger logger = LoggerFactory.getLogger(FGMM.class);
    // parameters of Gaussian
    private final DoubleMatrix muZero;
    private final double kappa;

    // parameters of Wishart
    private final DoubleMatrix sigmaZero;
    private final int nu;

    // parameters of Dirichlet
    private final double alpha;

    private final int dim;
    private final int nComponents;

    MersenneTwister mt = new MersenneTwister();

    public FGMM(int dim, int nComponents, DoubleMatrix muZero, double kappa, DoubleMatrix sigmaZero, int nu,
            double alpha) {
        this.dim = dim;
        this.nComponents = nComponents;
        this.muZero = muZero;
        this.kappa = kappa;
        this.sigmaZero = sigmaZero;
        this.nu = nu;
        this.alpha = alpha;
        logger.info("initialize the model");
        logger.info("dimention={}, #components={}", dim, nComponents);
        logger.info("mu_0={}, kappa={}, sigma_0={}, nu={}, alpha={}", muZero, kappa, sigmaZero, nu, alpha);
    }

    /**
     * Inferences with Gibbs sampling.
     * 
     * @param data observation data points. Array of row vectors
     * @param mu initial means of each component. Array of row vectors
     * @param sigma initial variance of each component
     * @param nIterations the number of iterations
     * @return ComponentParams a object containing estimated parameters of each component
     */
    public ComponentParams learn(DoubleMatrix[] data, DoubleMatrix[] mu, DoubleMatrix[] sigma, int nIterations) {
        logger.info("start inference");
        logger.info("#iterations={}", nIterations);
        logger.info("initial menas={}, initial variances={}", mu, sigma);
        if (data[0].rows != dim || mu[0].rows != dim || sigma[0].rows != sigma[0].columns || sigma[0].rows != dim) {
            throw new IllegalArgumentException("Dimention of the argument(s) is invalid.");
        }
        // initialize
        // latent variables specifying the mixture component of each observation
        int[] z = new int[data.length];
        // number of data points in each component.
        int[] n = new int[nComponents];

        // initialize z and n
        for (int i = 0; i < z.length; i++) {
            z[i] = mt.nextInt(nComponents);
            n[z[i]]++;
        }
        //likelihood
        double[] pi = computePi(n);
        logger.info("initial likelihood={}", computeLikelihood(data, pi, mu, sigma));
        
        // inference
        for (int iter = 0; iter < nIterations; iter++) {
            // sample and update z
            nAndz result = sampleLatentVariable(data, mu, sigma, n, z);
            n = result.n;
            z = result.z;

            // update the paramters
            for (int i = 0; i < nComponents; i++) {
                DoubleMatrix sampleMean = computeSampleMean(data, z, i);
                DoubleMatrix scatterMatrix = computeScatterMatrix(data, sampleMean, z, i);
                mu[i] = estimateMu(sampleMean, mu[i], n[i]);
                sigma[i] = estimateSigma(sampleMean, scatterMatrix, sigma[i], n[i]);
            }
            
            
            //likelihood
            pi = computePi(n);
            double likelihood = computeLikelihood(data, pi, mu, sigma);

            // logging
            if (iter % 10 == 0) {
                logger.info("iteration {} : menas={}, variances={}", iter, mu, sigma);
                logger.info("likelihood={}", likelihood);
            }

        }

   
        return new ComponentParams(mu, sigma, pi);
    }

    /**
     * Computes MAP estimate value of a mean parameter of a component.
     * 
     * @param sampleMean Sample mean of data points.
     * @param mu
     * @param n Number of data points in the component.
     * @return Estimated mu.
     */
    private DoubleMatrix estimateMu(DoubleMatrix sampleMean, DoubleMatrix mu, int n) {
        DoubleMatrix estimatedMu = mu.mul(kappa).add(sampleMean.mul(n)).mul(1.0 / (kappa + n));
        return estimatedMu;
    }

    /**
     * Computes MAP estimate value of sigma which is a variance parameter of a component.
     * 
     * @param sampleMean sample mean of data points in the component
     * @param scatterMatrix scatter matrix of data points in the component
     * @param sigma variance of the component
     * @param n number of data points
     * @return estimated sigma
     */
    private DoubleMatrix estimateSigma(DoubleMatrix sampleMean, DoubleMatrix scatterMatrix, DoubleMatrix sigma, int n) {
        // sampleMean - muZero
        DoubleMatrix sampleMeanMinusMuZero = sampleMean.sub(muZero);
        // (sampleMean-muZero)*(sampleMean-muZero)^T * {kappa*n/(kappa+n)}
        DoubleMatrix S = sampleMeanMinusMuZero.mmul(sampleMeanMinusMuZero.transpose()).mul(kappa * n / (kappa + n));
        DoubleMatrix delta = sigmaZero.add(scatterMatrix).add(S);

        DoubleMatrix estimatedSigma = delta.mul(1.0 / (kappa * (nu + n - dim + 1)));
        return estimatedSigma;
    }

    /**
     * Computes pi which is the mixture weight of the components.
     */
    private double[] computePi(int[] n) {
        double[] pi = new double[nComponents];

        int sum = 0;
        for (int i : n)
            sum += i;

        for (int i = 0; i < nComponents; i++) {
            pi[i] = n[i] / (double) sum;
        }
        return pi;
    }

    /**
     * Computes scatter matrix of data points in a component.
     * 
     * @param data
     * @param mu
     * @return
     */
    private DoubleMatrix computeScatterMatrix(DoubleMatrix[] data, DoubleMatrix sampleMeanInComponent, int[] z,
            int componentId) {
        DoubleMatrix scat = DoubleMatrix.zeros(dim, dim);
        for (int i = 0; i < data.length; i++) {
            if (z[i] == componentId) {
                DoubleMatrix vec = data[i].sub(sampleMeanInComponent);
                scat = scat.add(vec.mmul(vec.transpose()));
            }
        }
        return scat;
    }

    /**
     * Computes sample mean of data points in a component.
     * 
     * @param data
     */
    private DoubleMatrix computeSampleMean(DoubleMatrix[] data, int[] z, int componentId) {
        DoubleMatrix sampleMean = DoubleMatrix.zeros(dim);
        int count = 0;
        for (int i = 0; i < data.length; i++) {
            if (z[i] == componentId) {
                sampleMean = sampleMean.add(data[i]);
                count++;
            }
        }
        if(count==0){
            return sampleMean;
        }else{
        return sampleMean.mul(1.0 / count);
        }
    }

    /**
     * Computes likelihood
     */
    public double computeLikelihood(DoubleMatrix[] data, double[] pi, DoubleMatrix[] mu, DoubleMatrix[] sigma) {
        double likelihood = 0;
        for (int i = 0; i < data.length; i++) {
            double sum = 0;
            for (int k = 0; k < nComponents; k++) {
                double density = new MultivariateNormalDistribution(mu[k].toArray(), sigma[k].toArray2())
                        .density(data[i].toArray());
                sum += pi[k] * density;
            }
            likelihood += Math.log(sum);
        }
        return likelihood;

    }

    /**
     * Gibbs sampling for z.
     */
    private nAndz sampleLatentVariable(DoubleMatrix[] data, DoubleMatrix[] mu, DoubleMatrix[] sigma, int[] n, int[] z) {
        for (int i = 0; i < data.length; i++) {
            n[z[i]]--;
            double[] prob = new double[nComponents];
            for (int k = 0; k < nComponents; k++) {
                double gaussianDensity = new MultivariateNormalDistribution(mu[k].toArray(), sigma[k].toArray2())
                        .density(data[i].toArray());
                prob[k] = (n[k] + alpha / nComponents) / (data.length + alpha - 1) * gaussianDensity;
            }
            int sample = Util.sampleCategorical(prob);
            z[i] = sample;
            n[z[i]]++;
        }
        return new nAndz(n, z);
    }

    /**
     * Container for z and n.
     */
    private class nAndz {
        public nAndz(int[] n, int[] z) {
            this.n = n;
            this.z = z;
        }

        /** Number of observations in each component */
        private int[] n;

        /**
         * Latent variables specifying the mixture component of each observation.
         */
        private int[] z;
    }

    /**
     * Container for parameters of component distributions.
     */
    public class ComponentParams {
        public ComponentParams(DoubleMatrix[] mu, DoubleMatrix[] sigma, double[] pi) {
            this.mu = mu;
            this.sigma = sigma;
            this.pi = pi;
        }

        /** Mean paramters of Gaussian distributions. */
        public DoubleMatrix[] mu;
        /** Variance paramters of Gaussian distributions. */
        public DoubleMatrix[] sigma;
        /** Mixture weight of components. */
        public double[] pi;
    }
}
