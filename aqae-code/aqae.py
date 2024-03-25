# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2018, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Our Quantum Amplitude Estimation Algorithm."""

from __future__ import annotations
from typing import cast, Callable, Tuple
import warnings
import numpy as np
from scipy import stats
import math

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.primitives import BaseSampler, Sampler

from qiskit_algorithms.amplitude_estimators.amplitude_estimator import AmplitudeEstimator, AmplitudeEstimatorResult
from qiskit_algorithms.amplitude_estimators.estimation_problem import EstimationProblem
from qiskit_algorithms.exceptions import AlgorithmError

class OurAmplitudeEstimation(AmplitudeEstimator):
    r"""The Iterative Amplitude Estimation algorithm.

    This class implements the Iterative Quantum Amplitude Estimation (IQAE) algorithm, proposed
    in [1]. The output of the algorithm is an estimate that,
    with at least probability :math:`1 - \alpha`, differs by epsilon to the target value, where
    both alpha and epsilon can be specified.

    It differs from the original QAE algorithm proposed by Brassard [2] in that it does not rely on
    Quantum Phase Estimation, but is only based on Grover's algorithm. It improves on the Iterative 
    Quantum Amplitude Estimation (IQAE) algorithm propose proposed by Grinko etal [1].
    
    References:
        [1]: Grinko, D., Gacon, J., Zoufal, C., & Woerner, S. (2019).
             Iterative Quantum Amplitude Estimation.
             `arXiv:1912.05559 <https://arxiv.org/abs/1912.05559>`_.
        [2]: Brassard, G., Hoyer, P., Mosca, M., & Tapp, A. (2000).
             Quantum Amplitude Amplification and Estimation.
             `arXiv:quant-ph/0005055 <http://arxiv.org/abs/quant-ph/0005055>`_.
             
    This code is based on a copy of iae.py, which is part of the qiskit_algorithms package. See https://github.com/qiskit-community/qiskit-algorithms .
    """

    def __init__(
        self,
        epsilon_target: float,
        alpha: float,
        method: str = "standard",
        confint_method: str = "hoeffding",
        sampler: BaseSampler | None = None,
        debug: bool = False
    ) -> None:
        r"""
        The output of the algorithm is an estimate for the amplitude `a`, that with at least
        probability 1 - alpha has an error of epsilon. The number of A operator calls scales
        linearly in 1/epsilon.

        Args:
            epsilon_target: Target precision for estimation target `a`, has values between 0 and 0.5
            alpha: Confidence level, the target probability is 1 - alpha, has values between 0 and 1
            method: Can be "standard" for standard method with fixed number of iterations (default), or "accelerated"
                where number of iterations determined during execution.
            confint_method: Statistical method used to estimate the confidence intervals in
                each iteration. Can be 'hoeffding' for the Hoeffding intervals (default) or 'clopper_pearson'
                for Clopper-Pearson intervals or 'score' for score intervals.
            sampler: A sampler primitive to evaluate the circuits.

        Raises:
            AlgorithmError: if the method to compute the confidence intervals is not supported
            ValueError: If the target epsilon is not in (0, 0.5]
            ValueError: If alpha is not in (0, 1)
            ValueError: If confint_method is not supported
        """
        # validate ranges of input arguments
        if not 0 < epsilon_target <= 0.5:
            raise ValueError(f"The target epsilon must be in (0, 0.5], but is {epsilon_target}.")

        if not 0 < alpha < 1:
            raise ValueError(f"The confidence level alpha must be in (0, 1), but is {alpha}")

        if method not in {"standard", "accelerated"}:
            raise ValueError(
                f"The estimation method must be 'standard' or 'accelerated', but is {method}."
            )

        if confint_method not in {"hoeffding", "clopper_pearson", "score"}:
            raise ValueError(
                f"The confidence interval method must be 'hoeffding', 'clopper_pearson' or 'score', but is {confint_method}."
            )

        super().__init__()

        # store parameters
        self._epsilon = epsilon_target
        self._alpha = alpha
        self._method = method
        self._confint_method = confint_method
        self._sampler = sampler
        self._debug = debug
        
    @property
    def sampler(self) -> BaseSampler | None:
        """Get the sampler primitive.

        Returns:
            The sampler primitive to evaluate the circuits.
        """
        return self._sampler

    @sampler.setter
    def sampler(self, sampler: BaseSampler) -> None:
        """Set sampler primitive.

        Args:
            sampler: A sampler primitive to evaluate the circuits.
        """
        self._sampler = sampler

    @property
    def epsilon_target(self) -> float:
        """Returns the target precision ``epsilon_target`` of the algorithm.

        Returns:
            The target precision (which is half the width of the confidence interval).
        """
        return self._epsilon

    @epsilon_target.setter
    def epsilon_target(self, epsilon: float) -> None:
        """Set the target precision of the algorithm.

        Args:
            epsilon: Target precision for estimation target `a`.
        """
        self._epsilon = epsilon


    def construct_circuit(
        self, estimation_problem: EstimationProblem, k: int = 0, measurement: bool = False
    ) -> QuantumCircuit:
        r"""Construct the circuit :math:`\mathcal{Q}^k \mathcal{A} |0\rangle`.

        The A operator is the unitary specifying the QAE problem and Q the associated Grover
        operator.

        Args:
            estimation_problem: The estimation problem for which to construct the QAE circuit.
            k: The power of the Q operator.
            measurement: Boolean flag to indicate if measurements should be included in the
                circuits.

        Returns:
            The circuit implementing :math:`\mathcal{Q}^k \mathcal{A} |0\rangle`.
        """
        num_qubits = max(
            estimation_problem.state_preparation.num_qubits,
            estimation_problem.grover_operator.num_qubits,
        )
        circuit = QuantumCircuit(num_qubits, name="circuit")

        # add classical register if needed
        if measurement:
            c = ClassicalRegister(len(estimation_problem.objective_qubits))
            circuit.add_register(c)

        # add A operator
        circuit.compose(estimation_problem.state_preparation, inplace=True)

        # add Q^k
        if k != 0:
            circuit.compose(estimation_problem.grover_operator.power(k), inplace=True)

        # add optional measurement
        if measurement:
            # real hardware can currently not handle operations after measurements, which might
            # happen if the circuit gets transpiled, hence we're adding a safeguard-barrier
            circuit.barrier()
            circuit.measure(estimation_problem.objective_qubits, c[:])

        return circuit

    def _good_state_probability(
        self,
        problem: EstimationProblem,
        counts_dict: dict[str, int],
    ) -> tuple[int, float]:
        """Get the probability to measure '1' in the last qubit.

        Args:
            problem: The estimation problem, used to obtain the number of objective qubits and
                the ``is_good_state`` function.
            counts_dict: A counts-dictionary (with one measured qubit only!)

        Returns:
            #one-counts, #one-counts/#all-counts
        """
        one_counts = 0
        for state, counts in counts_dict.items():
            if problem.is_good_state(state):
                one_counts += counts

        return int(one_counts), one_counts / sum(counts_dict.values())
    
    def _compute_Ktheta(self, n, N, alpha, m) -> tuple[float, float, bool]:
        """Finds thetaflat and thetasharp, multiplied by latest value of K.

        Args:
            n: Number of successes.
            N: Number of trials.
            alpha: The confidence level.
            m: Number of rotations

        Returns:
            Value of L, new m and bool representing whether L and m could be found.
        """
        # start calculation of confidence intervals
        lower = 0
        upper = 1
        if self._confint_method == "hoeffding":
            lower, upper = _binomial_Hoeffding_ci(n, N, alpha)
        elif self._confint_method == "clopper_pearson":
            lower, upper = _binomial_CP_ci(n, N, alpha)
        elif self._confint_method == "score":
            lower, upper = _binomial_score_ci(n, N, alpha)

        # convert to angles
        Kthetaflat = np.arcsin(np.sqrt(lower))            
        Kthetasharp = np.arcsin(np.sqrt(upper))       

        #make an adjustment so that Kthetaflat and Kthetasharp are both between lower and upper bound
        Kthetaflat = _adjust_angle (Kthetaflat, m)
        Kthetasharp = _adjust_angle (Kthetasharp, m)
        
        return min(Kthetaflat,Kthetasharp), max(Kthetaflat,Kthetasharp)

    def _compute_L_m(self, Theta_flat, Theta_sharp, m0, error = 1e-10) -> tuple[float, float, bool]:
        """Finds L and m.

        Args:
            Theta_flat: lower boundary of copnfidence interval, times latest value of K
            Theta_sharp: upper boundary of confidence interval, times latest value of K.
            m0: current value of m.
            error: Tolerance level (used to deal with numerical errors).

        Returns:
            Value of L, new m and bool representing whether L and m could be found.

        Raises:
            AlgorithmError: Sampler job run error.
        """
        found=False
        halfpi = np.pi/2
        for L in [3,5,7]:
            for m1 in range(L*m0,L*m0+L):
                #if self._debug:
                #    print (L, m1, m1*halfpi - error, "<=", L*Theta_flat, "<=", L*Theta_sharp + error, "<=", (m1+1)*halfpi, ":", m1*halfpi - error <= L*Theta_flat and L*Theta_sharp <= (m1+1)*halfpi + error)
                if m1*halfpi - error <= L*Theta_flat and L*Theta_sharp <= (m1+1)*halfpi + error:
                    found=True
                    break
            else:         #method for breaking nested for loops
                continue  #continue pertains to outer loop
            break

        return L, m1, found

    def _estimate_standard(
        self, estimation_problem: EstimationProblem
    ) -> "OurAmplitudeEstimationResult":
        """Run the standard amplitude estimation algorithm on provided estimation problem.

        Args:
            estimation_problem: The estimation problem.

        Returns:
            An amplitude estimation results object.

        Raises:
            ValueError: A Sampler must be provided.
            AlgorithmError: Sampler job run error, or error in finding L and m.
        """
        
        if self._sampler is None:
            warnings.warn("No sampler provided, defaulting to Sampler from qiskit.primitives")
            self._sampler = Sampler()

        # useful constants
        halfpi = np.pi/2
        E = 0.5*(np.sin(3*np.pi/14)**2 - np.sin(np.pi/6)**2)
        F = 0.5*np.arcsin(np.sqrt(2*E))
        C = 4*F/(6*F+np.pi)
        J = np.emath.logn(3, 9*F/self._epsilon)
        if self._debug:
            print("epsilon =", self._epsilon)
            print("E =",E)
            print("F =",F)
            print("C =",C)
            print("J =",J)
        
        # initialize variables
        k = []  # list of powers k: Q^k
        alpha = [] # list of confidence levels
        K = [1] 
        m = [0]
        N = []
        n = []
        A = []
        L = []
        theta_intervals = []  # confidence intervals for theta (called ['theta_flat', 'theta_sharp'] in paper)
        a_intervals = []  # a priori knowledge of the confidence interval of the estimate - calculate in one shot at end?
        
        num_oracle_queries = 0 # number of calls to Q

        # ====================================
        # main loop goes here
        
        num_iterations = -1  # called 'i' in paper
        
        #terminal condition - loop will run at least once
        finished = False
        
        while not finished:
            num_iterations += 1
            
            alpha.append(C*self._alpha*self._epsilon*K[-1]/F)
            N.append(math.ceil(0.5*np.log(2*J/alpha[-1])/E**2))
            k.append(int((K[-1]-1)/2))
            
            if self._debug:
                print("============================================")
                print("i =",num_iterations)
                print("alpha_i =",alpha[-1])
                print("N_i =",N[-1])
                print("K_i =",K[-1])
                print("k_i =",k[-1])
            
            # run measurements for Q^k A|0> circuit
            circuit = self.construct_circuit(estimation_problem, k[-1], measurement = True)
            
            try:
                job = self._sampler.run([circuit], shots = N[-1])
                ret = job.result()
            except Exception as exc:
                raise AlgorithmError("The job was not completed successfully.") from exc
            
            # retrieve counts of outcomes
            counts = { k: round(v * N[-1]) for k, v in ret.quasi_dists[0].binary_probabilities().items() }
            
            # calculate the probability of measuring '1', 'prob' is \hat{A}_i in the paper
            one_counts, prob = self._good_state_probability(estimation_problem, counts)
            n.append(one_counts)
            A.append(prob)
            if self._debug:
                print("n_i =",n[-1])
                print("A_i =",A[-1])

            # track number of Q-oracle calls
            num_oracle_queries += N[-1] * k[-1]
            
            # determine confidence intervals
            Kthetaflat, Kthetasharp = self._compute_Ktheta(n[-1], N[-1], alpha[-1], m[-1])
            
            thetaflat = Kthetaflat/K[-1]
            thetasharp = Kthetasharp/K[-1]
            if self._debug:
                print("thetaflat_i =",thetaflat)
                print("thetasharp_i =",thetasharp)
            
            #termination condition can be determined now
            finished = thetasharp-thetaflat <= 2*self._epsilon
            
            #save confidence intervals
            theta_intervals.append([thetaflat,thetasharp])
            a_intervals.append([np.sin(theta_intervals[-1][0])**2,np.sin(theta_intervals[-1][1])**2])

            if not finished:
                try:
                    newL, newm, found = self._compute_L_m(Kthetaflat,Kthetasharp,m[-1])
                    L.append(newL)
                    m.append(newm)
                    K.append(K[-1]*newL)
                    if self._debug:
                        print("L_i =",L[-1])
                        print("m_i =",m[-1])
                except Exception as exc:
                    #cannot find L or m. Redo this round
                       
                    num_iterations -= 1
                    del alpha[-1]
                    del N[-1]
                    del k[-1]
                    del n[-1]
                    del A[-1]
                    del theta_intervals[-1]
                    del a_intervals[-1]
                    
                    raise AlgorithmError("Cannot find L or m, likely due to rounding error.") from exc
                            
        # main loop finishes here
        # ====================================

        # get the latest confidence interval for the estimate of a
        confidence_interval = cast(Tuple[float, float], a_intervals[-1])

        # the final estimate is the mean of the confidence interval
        estimation = np.mean(confidence_interval)

        result = OurAmplitudeEstimationResult()
        result.alpha = self._alpha
        result.post_processing = cast(Callable[[float], float], estimation_problem.post_processing)
        result.num_oracle_queries = num_oracle_queries

        result.estimation = float(estimation)
        result.epsilon_estimated = (confidence_interval[1] - confidence_interval[0]) / 2
        result.confidence_interval = confidence_interval

        result.estimation_processed = estimation_problem.post_processing(
            estimation  # type: ignore[arg-type,assignment]
        )
        confidence_interval = tuple(
            estimation_problem.post_processing(x)  # type: ignore[arg-type,assignment]
            for x in confidence_interval
        )

        result.confidence_interval_processed = confidence_interval
        result.epsilon_estimated_processed = (confidence_interval[1] - confidence_interval[0]) / 2
        result.estimate_intervals = a_intervals
        result.theta_intervals = theta_intervals

        result.k = k
        result.alpha = alpha
        result.K = K 
        result.m = m
        result.N = N
        result.n = n
        result.A = A
        result.L = L
        
        return result

    def _estimate_accelerated(
        self, estimation_problem: EstimationProblem
    ) -> "OurAmplitudeEstimationResult":
        """Run the accelerated amplitude estimation algorithm on provided estimation problem.

        Args:
            estimation_problem: The estimation problem.

        Returns:
            An amplitude estimation results object.

        Raises:
            ValueError: A Sampler must be provided.
            AlgorithmError: Sampler job run error.
        """
        
        if self._sampler is None:
            warnings.warn("No sampler provided, defaulting to Sampler from qiskit.primitives")
            self._sampler = Sampler()

        # useful constants
        halfpi = np.pi/2
        # E = 0.5*(np.sin(3*np.pi/14)**2 - np.sin(np.pi/6)**2)
        # F = 0.5*np.arcsin(np.sqrt(2*E))
        C = 8/3/np.pi
        if self._debug:
            print("epsilon =", self._epsilon)
            print("C =",C)
            
        # initialize variables
        k = []  # list of powers k: Q^k
        alpha = [] # list of confidence levels
        K = [1] 
        m = [0]
        alpha = []
        N = []
        n = []
        A = []
        L = []
        theta_intervals = []  # confidence intervals for theta (called ['theta_flat', 'theta_sharp'] in paper)
        a_intervals = []  # a priori knowledge of the confidence interval of the estimate - calculate in one shot at end?
        
        num_oracle_queries = 0 # number of calls to Q

        # ====================================
        # main loop goes here
        
        num_iterations = -1  # called 'i' in paper
        
        #terminal condition - loop will run at least once
        finished = False
        
        while not finished:
            num_iterations += 1
            
            #values
            alpha.append(C*self._alpha*self._epsilon*K[-1])
            k.append(int((K[-1]-1)/2))
            
            #create spaces for variables
            N.append(0)
            n.append(0)
            A.append(np.nan)
            
            if self._debug:
                print("============================================")
                print("i =",num_iterations)
                print("alpha_i =",alpha[-1])
                print("K_i =",K[-1])
                print("k_i =",k[-1])
            
            # run measurements for Q^k A|0> circuit
            circuit = self.construct_circuit(estimation_problem, k[-1], measurement = True)
            
            found = False
            while not found:
            
                # run measurements for Q^k A|0> circuit
                try:
                    job = self._sampler.run([circuit], shots = 1)
                    ret = job.result()
                except Exception as exc:
                    raise AlgorithmError("The job was not completed successfully. ") from exc

                # retrieve counts of outcomes
                num_shots = ret.metadata[0].get("shots")
                N[-1] += num_shots
                counts = { k: round(v)*num_shots for k, v in ret.quasi_dists[0].binary_probabilities().items() }

                # calculate the probability of measuring '1', 'prob' is \hat{A}_i in the paper
                one_counts, prob = self._good_state_probability(estimation_problem, counts)
                n[-1] += one_counts
                A[-1] = n[-1] / N[-1]

                # determine confidence intervals
                Kthetaflat, Kthetasharp = self._compute_Ktheta(n[-1], N[-1], alpha[-1], m[-1])
                
                newL, newm, found = self._compute_L_m(Kthetaflat,Kthetasharp,m[-1])
                if found:
                    L.append(newL)
                    m.append(newm)
                    K.append(K[-1]*newL)
                    if self._debug:
                        print("L_i =",L[-1])
                        print("m_i =",m[-1])

            # track number of Q-oracle calls
            num_oracle_queries += N[-1] * k[-1]

            # calculate estimates
            thetaflat = Kthetaflat/K[-2]
            thetasharp = Kthetasharp/K[-2]
            if self._debug:
                print("thetaflat_i =",thetaflat)
                print("thetasharp_i =",thetasharp)

            # save confidence intervals
            theta_intervals.append([min(thetaflat,thetasharp),max(thetaflat,thetasharp)])
            a_intervals.append([np.sin(theta_intervals[-1][0])**2,np.sin(theta_intervals[-1][1])**2])

            #termination condition can be determined now
            finished = abs(thetasharp-thetaflat) <= 2*self._epsilon
            
            if self._debug:
                print("n_i =",n[-1])
                print("N_i =",N[-1])
                print("A_i =",A[-1])
                    

        # main loop finishes here
        # ====================================

        # get the latest confidence interval for the estimate of a
        confidence_interval = cast(Tuple[float, float], a_intervals[-1])

        # the final estimate is the mean of the confidence interval
        estimation = np.mean(confidence_interval)

        result = OurAmplitudeEstimationResult()
        result.alpha = self._alpha
        result.post_processing = cast(Callable[[float], float], estimation_problem.post_processing)
        result.num_oracle_queries = num_oracle_queries

        result.estimation = float(estimation)
        result.epsilon_estimated = (confidence_interval[1] - confidence_interval[0]) / 2
        result.confidence_interval = confidence_interval

        result.estimation_processed = estimation_problem.post_processing(
            estimation  # type: ignore[arg-type,assignment]
        )
        confidence_interval = tuple(
            estimation_problem.post_processing(x)  # type: ignore[arg-type,assignment]
            for x in confidence_interval
        )

        result.confidence_interval_processed = confidence_interval
        result.epsilon_estimated_processed = (confidence_interval[1] - confidence_interval[0]) / 2
        result.estimate_intervals = a_intervals
        result.theta_intervals = theta_intervals

        result.k = k
        result.alpha = alpha
        result.K = K 
        result.m = m
        result.N = N
        result.n = n
        result.A = A
        result.L = L
        
        return result

    def estimate(
        self, estimation_problem: EstimationProblem
    ) -> "OurAmplitudeEstimationResult":
        """Run the amplitude estimation algorithm on provided estimation problem.

        Args:
            estimation_problem: The estimation problem.

        Returns:
            An amplitude estimation results object.

        Raises:
            ValueError: A Sampler must be provided.
            AlgorithmError: Sampler job run error.
        """
        
        if self._method == "standard":
            return self._estimate_standard(estimation_problem)
        else:
            return self._estimate_accelerated(estimation_problem)
        
 
class OurAmplitudeEstimationResult(AmplitudeEstimatorResult):
    """The ``OurAmplitudeEstimation`` result object."""

    def __init__(self) -> None:
        super().__init__()
        self._alpha: float | None = None
        self._epsilon_target: float | None = None
        self._epsilon_estimated: float | None = None
        self._epsilon_estimated_processed: float | None = None
        Estimate_intervals: list[list[float]] | None = None
        self._theta_intervals: list[list[float]] | None = None
        self._powers: list[int] | None = None
        self._ratios: list[float] | None = None
        Confidence_interval_processed: tuple[float, float] | None = None

    @property
    def alpha(self) -> list[float]:
        r"""Return the confidence level :math:`\alpha`."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: list[float]) -> None:
        r"""Set the confidence level :math:`\alpha`."""
        self._alpha = value

    @property
    def epsilon_target(self) -> float:
        """Return the target half-width of the confidence interval."""
        return self._epsilon_target

    @epsilon_target.setter
    def epsilon_target(self, value: float) -> None:
        """Set the target half-width of the confidence interval."""
        self._epsilon_target = value

    @property
    def epsilon_estimated(self) -> float:
        """Return the estimated half-width of the confidence interval."""
        return self._epsilon_estimated

    @epsilon_estimated.setter
    def epsilon_estimated(self, value: float) -> None:
        """Set the estimated half-width of the confidence interval."""
        self._epsilon_estimated = value

    @property
    def epsilon_estimated_processed(self) -> float:
        """Return the post-processed estimated half-width of the confidence interval."""
        return self._epsilon_estimated_processed

    @epsilon_estimated_processed.setter
    def epsilon_estimated_processed(self, value: float) -> None:
        """Set the post-processed estimated half-width of the confidence interval."""
        self._epsilon_estimated_processed = value

    @property
    def estimate_intervals(self) -> list[list[float]]:
        """Return the confidence intervals for the estimate in each iteration."""
        return Estimate_intervals

    @estimate_intervals.setter
    def estimate_intervals(self, value: list[list[float]]) -> None:
        """Set the confidence intervals for the estimate in each iteration."""
        Estimate_intervals = value

    @property
    def theta_intervals(self) -> list[list[float]]:
        """Return the confidence intervals for the angles in each iteration."""
        return self._theta_intervals

    @theta_intervals.setter
    def theta_intervals(self, value: list[list[float]]) -> None:
        """Set the confidence intervals for the angles in each iteration."""
        self._theta_intervals = value

    @property
    def k(self) -> list[int]:
        """Return the powers of the Grover operator in each iteration."""
        return self._k

    @k.setter
    def k(self, value: list[int]) -> None:
        """Set the power of the Grover operator in each iteration."""
        self._k = value

    @property
    def K(self) -> list[float]:
        r"""Return :math:`K_{i}` for each iteration :math:`i`."""
        return self._K

    @K.setter
    def K(self, value: list[float]) -> None:
        r"""Set :math:`K_{i}` for each iteration :math:`i`."""
        self._K = value

    @property
    def m(self) -> list[float]:
        r"""Return :math:`m_{i}` for each iteration :math:`i`."""
        return self._m

    @m.setter
    def m(self, value: list[float]) -> None:
        r"""Set :math:`m_{i}` for each iteration :math:`i`."""
        self._m = value

    @property
    def N(self) -> list[float]:
        r"""Return :math:`N_{i}` for each iteration :math:`i`."""
        return self._N

    @N.setter
    def N(self, value: list[float]) -> None:
        r"""Set :math:`N_{i}` for each iteration :math:`i`."""
        self._N = value

    @property
    def n(self) -> list[float]:
        r"""Return :math:`n_{i}` for each iteration :math:`i`."""
        return self._n

    @n.setter
    def n(self, value: list[float]) -> None:
        r"""Set :math:`n_{i}` for each iteration :math:`i`."""
        self._n = value

    @property
    def A(self) -> list[float]:
        r"""Return :math:`A_{i}` for each iteration :math:`i`."""
        return self._A

    @A.setter
    def A(self, value: list[float]) -> None:
        r"""Set :math:`A_{i}` for each iteration :math:`i`."""
        self._A = value

    @property
    def L(self) -> list[float]:
        r"""Return :math:`L_{i}` for each iteration :math:`i`."""
        return self._L

    @L.setter
    def L(self, value: list[float]) -> None:
        r"""Set :math:`L_{i}` for each iteration :math:`i`."""
        self._L = value

    @property
    def confidence_interval_processed(self) -> tuple[float, float]:
        """Return the post-processed confidence interval."""
        return Confidence_interval_processed

    @confidence_interval_processed.setter
    def confidence_interval_processed(self, value: tuple[float, float]) -> None:
        """Set the post-processed confidence interval."""
        Confidence_interval_processed = value

#given value of theta and n, finds value kappa such that |sin theta| = |sin kappa| and n*halfpi <= kappa <= (n+1)*halfpi
#we assume n>=0 so only adjustments to the right are considered
def _adjust_angle (theta: float, n: int):
    halfpi = np.pi/2
    
    lowerbound = n*halfpi
    upperbound = lowerbound+halfpi

    if theta < lowerbound:
        #adjust up by full periods of |sin|
        adjust = np.floor((upperbound - theta)/np.pi)
        if adjust > 0:
            theta += adjust*np.pi

        #now small adjustment if needed
        if theta < lowerbound:
            theta = 2*lowerbound - theta

    return theta

# Returns Hoeffding confidence interval
def _binomial_Hoeffding_ci(n, N, alpha) -> tuple[float, float]:
    """Compute the Hoeffding confidence interval for `N` i.i.d. Bernoulli trials.

    Args:
        n: Number of successes.
        N: Number of trials.
        alpha: The confidence level.

    Returns:
        The Hoeffding confidence interval.
    """
    Epsilon=np.sqrt((np.log(2/alpha))/(2*N))
    p_min=max(n/N-Epsilon,0.0)
    p_max=min(n/N+Epsilon,1.0)
    return p_min, p_max

# Returns Clopper-Pearson confidence interval
def _binomial_CP_ci(n, N, alpha):
    """Compute the Clopper-Pearson confidence interval for `N` i.i.d. Bernoulli trials.

    Args:
        n: Number of successes.
        N: Number of trials.
        alpha: The confidence level.

    Returns:
        The Clopper-Pearson confidence interval.
    """
    if n==0:
        p_min=0.0
        p_max=stats.beta.interval(1-alpha, n+1,N-n)[1]
    elif n==N:
        p_min=stats.beta.interval(1-alpha, n,N-n+1)[0]
        p_max=1.0
    else:
        p_min=stats.beta.interval(1-alpha, n,N-n+1)[0]
        p_max=stats.beta.interval(1-alpha, n+1,N-n)[1]
    return p_min, p_max

# Returns the score confidence interval; see Agresti & Coull (1998) 
def _binomial_score_ci(n, N, alpha):
    """Compute the score confidence interval for `N` i.i.d. Bernoulli trials.
    See Agresti & Coull (1998).

    Args:
        n: Number of successes.
        N: Number of trials.
        alpha: The confidence level.

    Returns:
        The score confidence interval.
    """
    z_a2=stats.norm.ppf(1-alpha/2)
    hat_p=n/N
    p_min=(hat_p+z_a2**2/(2*N)-z_a2*np.sqrt((hat_p*(1-hat_p)+z_a2**2/(4*N))/N))/(1+z_a2**2/N)
    # preventing p_min<0.0 or p_max>1.0 due to rounding errors 
    if p_min<0.0:
        p_min=0.0
    p_max=(hat_p+z_a2**2/(2*N)+z_a2*np.sqrt((hat_p*(1-hat_p)+z_a2**2/(4*N))/N))/(1+z_a2**2/N)
    if p_max>1.0:
        p_max=1.0
    return p_min, p_max