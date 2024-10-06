import random

from deap import base
from deap import creator
from deap import tools
from copy import deepcopy
from lesson import LessonType, Lesson
from functools import partial

#input for CSP
teachers_info = {"teacher1": {"subjects": ["subj1"], "pairs": 7}, #pairs per week
            "teacher2": {"subjects": ["subj4"], "pairs": 5}
            }

groups_info = {"group1": {"subj1": {"lectures_count": 2, "practices_count": 2 }, #lect|pract per week
                     "subj4": {"lectures_count": 1, "practices_count": 2 }
                     },
          "group2": {"subj1": {"lectures_count": 2, "practices_count": 3 },
                     "subj4": {"lectures_count": 1, "practices_count": 2 }
                     }
         }


#penalties = {"teacher_pairs_count": 10000, #equals
#             "teacher_same_time_pairs": 100000,
#             "group_subject_pairs_count": 10000, #strictly equals. for each group and for each subject.
#                                                # for both lectures and practices.
#             "group_same_time_pairs": 100000,
#             "more_than_one_group_at_practice": 1000,
#             "less_than_two_groups_at_lecture": 1000,
#             "teacher_inappropriate_subject": 100000,
#             "group_inappropriate_subject": 100000,
#             }
#
#donuts = {"optimal_pairs_count": 100, # 3 or 0 pairs per day
#          "no_fourth_pair": 100}

#domain_base
def create_domain(teachers_info, groups_info):
    domain = dict()
    domain["subjects"] = list( set([s for t in teachers_info.values() for s in t["subjects"]]) )
    domain["teachers"] = list(teachers_info.keys())
    domain["groups"] = list(groups_info.keys())
    domain["pair_types"] = list(LessonType)
    domain["pair_time"] = [1,2,3,4]
    domain["working_days"] = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    return domain

#constraints:
def c_teachers_pairs_count(domain, assignment):
    teachers_pairs_count = dict()
    penalties_count = 0

    for t in domain["teachers"]: teachers_pairs_count[t] = 0

    for dayData in assignment:
        for lesson in dayData:
            teachers_pairs_count[lesson.teacher] += 1
    for t in teachers_pairs_count.keys():
        if teachers_pairs_count[t] != teachers_info[t]["pairs"]:
            penalties_count += 1

    return penalties_count, 10000 #penalty for each of violations
def c_teacher_same_time_pairs(domain, assignment):
    penalties_count = 0
    for dayData in assignment:
        teachers_pairs_count = dict()
        for t in domain["teachers"]:
            teachers_pairs_count[t] = dict()
            for i in domain["pair_time"]: teachers_pairs_count[t][f"{i}"] = 0

        for lesson in dayData:
            teachers_pairs_count[lesson.teacher][f"{lesson.lessonNum}"] += 1

        for t in teachers_pairs_count.keys():
            for i in domain["pair_time"]:
                if teachers_pairs_count[t][f"{i}"] > 1:
                    penalties_count += 1

    return penalties_count, 100000
def c_group_subject_pairs_count(domain, assignment):
    penalties_count = 0
    groups_week_pairs_count = dict()
    for g in domain["groups"]:
        groups_week_pairs_count[g] = dict()
        for s in groups_info[g].keys():
            groups_week_pairs_count[g][s] = dict()
            groups_week_pairs_count[g][s]["lectures_count"] = 0
            groups_week_pairs_count[g][s]["practices_count"] = 0

    for dayData in assignment:
        for lesson in dayData:
            for g in lesson.groups:
                if lesson.type == LessonType.Lecture:
                    groups_week_pairs_count[g][lesson.subject]["lectures_count"] += 1
                elif lesson.type == LessonType.Practice:
                    groups_week_pairs_count[g][lesson.subject]["practices_count"] += 1

    for g in domain["groups"]:
        for s in groups_info[g].keys():
            if groups_week_pairs_count[g][s]["lectures_count"] != groups_info[g][s]["lectures_count"]:
                penalties_count += 1
            if groups_week_pairs_count[g][s]["practices_count"] != groups_info[g][s]["practices_count"]:
                penalties_count += 1

    return penalties_count, 10000
def c_group_same_time_pairs(domain, assignment):
    penalties_count = 0
    for dayData in assignment:
        groups_pairs_count = dict()
        for g in domain["groups"]:
            groups_pairs_count[g] = dict()
            for i in domain["pair_time"]: groups_pairs_count[g][f"{i}"] = 0

        for lesson in dayData:
            for g in lesson.groups:
                groups_pairs_count[g][f"{lesson.lessonNum}"] += 1

        for g in groups_pairs_count.keys():
            for i in domain["pair_time"]:
                if groups_pairs_count[g][f"{i}"] > 1:
                    penalties_count += 1

    return penalties_count, 100000
def c_teacher_inappropriate_subject(domain, assignment):
    penalties_count = 0
    for dayData in assignment:
        for lesson in dayData:
            if lesson.subject not in teachers_info[lesson.teacher]["subjects"]:
                penalties_count += 1
    return penalties_count, 100000
def c_group_inappropriate_subject(domain, assignment):
    penalties_count = 0
    for dayData in assignment:
        for lesson in dayData:
            for g in lesson.groups:
                if lesson.subject not in groups_info[g].keys():
                    penalties_count += 1
    return penalties_count, 100000
def c_more_than_one_group_at_practice(domain, assignment):
    penalties_count = 0
    for dayData in assignment:
        for lesson in dayData:
            if lesson.type == LessonType.Practice and len(lesson.groups) > 1:
                    penalties_count += 1

    return penalties_count, 1000
def c_less_than_two_groups_at_lecture(domain, assignment):
    penalties_count = 0
    for dayData in assignment:
        for lesson in dayData:
            if lesson.type == LessonType.Lecture and len(lesson.groups) < 2:
                penalties_count += 1

    return penalties_count, 1000
def create_constraints():
    constraints = list()
    constraints.append(c_teachers_pairs_count)
    constraints.append(c_teacher_same_time_pairs)
    constraints.append(c_group_subject_pairs_count)
    constraints.append(c_group_same_time_pairs)
    constraints.append(c_teacher_inappropriate_subject)
    constraints.append(c_group_inappropriate_subject)
    constraints.append(c_more_than_one_group_at_practice)
    constraints.append(c_less_than_two_groups_at_lecture)
    return constraints

#donuts:
def d_optimal_pairs_count(domain, assignment): #0 or 3
    donuts_count = 0
    for dayData in assignment:
        teachers_pairs_count = dict()
        for t in domain["teachers"]: teachers_pairs_count[t] = 0

        groups_pais_count = dict()
        for g in domain["groups"]: groups_pais_count[g] = 0

        for lesson in dayData:
            teachers_pairs_count[lesson.teacher] += 1
            for g in lesson.groups:
                groups_pais_count[g] += 1

        for t in teachers_pairs_count.values():
            if t in [0, 3]:
                donuts_count += 1

        for g in groups_pais_count.values():
            if g in [0, 3]:
                donuts_count += 1
    return donuts_count, 100
def d_no_fourth_pair(domain, assignment):
    donuts_count = 0
    for dayData in assignment:
        groups_pairs_count = dict()
        for g in domain["groups"]:
            groups_pairs_count[g] = dict()
            for i in domain["pair_time"]: groups_pairs_count[g][f"{i}"] = 0

        for lesson in dayData:
            for g in lesson.groups:
                groups_pairs_count[g][f"{lesson.lessonNum}"] += 1

        for g in groups_pairs_count.keys():
            for i in domain["pair_time"]:
                if groups_pairs_count[g]["4"] == 0:
                    donuts_count += 1

    return donuts_count, 100
def create_donuts():
    donuts = list()
    donuts.append(d_optimal_pairs_count)
    donuts.append(d_no_fourth_pair)
    return donuts

#chromosome generation
def generate_lesson(domain):
    lesson_type = random.choice(domain["pair_types"])
    subject = random.choice(domain["subjects"])
    teacher = random.choice(domain["teachers"])
    groups_count = random.randint(1, len(domain["groups"]))
    groups_arr = random.sample(domain["groups"], groups_count)
    lesson_num = random.choice(domain["pair_time"])

    return Lesson(lesson_type=lesson_type,
                  teacher=teacher,
                  groups=groups_arr,
                  subject=subject,
                  lesson_num= lesson_num)
def generate_day_schedule(domain):
    pairs_count = random.randint(0, 4*len(domain["groups"]) )
    return [generate_lesson(domain) for _ in range(pairs_count)]
def generate_ind(domain):
    ind = creator.Individual()
    for _ in range(len(domain["working_days"])):
        ind.append(generate_day_schedule(domain))
    return ind

#GA COP(CSP) cycle functions
def objective(domain, constraints, donuts, assignment): #assignment = individual = chromosome
    score = 0
    for c in constraints:
        penalties_count, penalty_value = c(domain, assignment)
        score -= penalties_count * penalty_value

    for d in donuts:
        donuts_count, donut_value = d(domain, assignment)
        score += donuts_count * donut_value
    return score
def heuristics_GA(population):
    return tools.selTournament(population, len(population), tournsize=3)
def heuristics_GA2(population):
    best_fit_count = 10 #save n best individuals for future, the rest needs to be changed.
    best_fit = max(population, key=lambda ind: ind.fitness.values[0])

    offspring = [best_fit] * best_fit_count
    offspring.extend(tools.selRandom(individuals=population, k=len(population)-best_fit_count ))

    return offspring
#def heuristics_GA3(population):

def crossover_func(ind1, ind2):
    return tools.cxTwoPoint(ind1, ind2)
def mutation_func(domain, ch_d_pb, ch_l_pb, ind):
    for i in range(len(ind)):
        if random.random() < ch_d_pb:
            ind[i] = generate_day_schedule(domain)
        for lesson_id in range(len(ind[i])):
            if random.random() < ch_l_pb:
                ind[i][lesson_id] = generate_lesson(domain)
    return ind


#solver
def GA(iters_count):
    domain = create_domain(teachers_info, groups_info)
    constraints = create_constraints()
    donuts = create_donuts()

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # fitness is time needed for machines to finish all tasks
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", generate_ind, domain=domain)
    toolbox.register("population", tools.initRepeat, container=list, func=toolbox.individual)

    toolbox.register("evaluate", partial(objective, domain, constraints, donuts))
    toolbox.register("mate", crossover_func)
    toolbox.register("mutate", mutation_func, domain, 0.55, 0.6) #domain, ch_d_pb, ch_l_pb
    toolbox.register("select", heuristics_GA2)



    pop = toolbox.population(n=400)

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = (fit,)

    CXPB = 0.33  #is the probability with which two individuals are crossed
    MUTPB = 0.44 # is the probability for mutating an individual

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0
    # Begin the evolution
    while g < iters_count:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)
        # Select the next generation individuals
        offspring = toolbox.select(pop)
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)

        min_ind = min(offspring, key= lambda i: i.fitness.values[0])
        max_ind = max(offspring, key=lambda i: i.fitness.values[0])
        print ("best ind fitness:", max_ind.fitness.values[0])

        print("worst ind fitness:", min_ind.fitness.values[0])

        pop[:] = offspring
    sortedPop = sorted(pop, key= lambda x: x.fitness.values[0], reverse=True)

    return sortedPop[0]


def print_sol(domain, ind):
    for i, day in enumerate(domain["working_days"].values()):
        print(f"{day} lessons:")
        for lesson in ind[i]:
            print(f"type: {lesson.type}, subject: {lesson.subject}, teacher: {lesson.teacher}, groups: {lesson.groups}, pair: {lesson.lessonNum}")
    print("fitness achieved:", ind.fitness.values[0])
    print()
    print()

def solve_COP():
    iters_count = 50000

    solution = GA(iters_count)
    print(f"solution:")
    print_sol(create_domain(teachers_info, groups_info), solution)

if __name__ == "__main__":
    solve_COP()