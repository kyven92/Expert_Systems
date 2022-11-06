# Expert_Systems


Process
1. Create an initial population of P chromosomes (generation 0).
2. Evaluate the fitness of each chromosome.
3. Select P parents from the current population via proportional selection (i.e.,the selection probability is proportional to the fitness).
4. Choose at random a pair of parents for mating. Exchange bit strings with the one-point crossover to create two offspring.
5. Process each offspring by the mutation operator, and insert the resulting offspring in the new population.
6. Repeat steps 4 and 5 until all parents are selected and mated (P offspring are created).
7. Replace the old population of chromosomes by the new one.
8. Evaluate the fitness of each chromosome in the new population.
9. Go back to step 3 if the number of generations is less than some upper bound. Otherwise, the final result is the best chromosome created during the search. 