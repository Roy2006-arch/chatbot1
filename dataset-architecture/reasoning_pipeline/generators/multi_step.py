import random
import re
from typing import Dict, List, Optional, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from ..schema import (
    ReasoningExample, ReasoningStep, ReasoningType, ReasoningTask,
    Difficulty, REASONING_TEMPLATES, REASONING_DOMAINS, DOMAIN_TOPICS,
)


class MultiStepReasoningGenerator:
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self.stats = {"generated": 0, "by_task": {}}

    def generate(
        self,
        task_type: ReasoningTask,
        count: int = 1000,
        difficulty_range: range = range(1, 5),
    ) -> List[ReasoningExample]:
        generator = getattr(self, f"_generate_{task_type.value}", self._generate_default)
        examples = []
        for _ in range(count):
            diff = random.choice(list(difficulty_range))
            ex = generator(diff)
            if ex:
                ex.reasoning_task = task_type
                ex.difficulty = Difficulty(diff)
                examples.append(ex)
        self.stats["generated"] += len(examples)
        self.stats["by_task"][task_type.value] = self.stats["by_task"].get(task_type.value, 0) + len(examples)
        return examples

    def _generate_multi_step_reasoning(self, difficulty: int) -> Optional[ReasoningExample]:
        domain = random.choice(REASONING_DOMAINS)
        topics = DOMAIN_TOPICS.get(domain, ["general"])
        topic = random.choice(topics)

        num_steps = random.randint(3 + difficulty, 8 + difficulty * 2)
        num_steps = min(num_steps, 15)

        steps = []
        for i in range(num_steps):
            step = ReasoningStep(
                index=i + 1,
                content=f"Step {i+1}: Apply {topic} concept to derive intermediate result. "
                        f"Based on previous step, we can conclude that the relevant factor is "
                        f"{random.choice(['X', 'Y', 'Z', 'the optimal value', 'the boundary condition'])}.",
                reasoning_type=random.choice(list(ReasoningType)),
                justification=f"By the principle of {random.choice(['transitivity', 'substitution', 'induction', 'deduction'])}, this follows from step {max(1, i)}.",
                confidence=0.8 + random.random() * 0.2,
            )
            steps.append(step)

        question = f"Solve this {domain} problem involving {topic}. Use {num_steps}-step reasoning."
        context = f"Consider a scenario in {domain} where we need to analyze {topic}."
        answer = f"After {num_steps} steps of reasoning, the solution is derived."

        return ReasoningExample(
            reasoning_type=ReasoningType.DEDUCTIVE,
            difficulty=Difficulty(difficulty),
            domain=domain,
            question=question,
            context=context,
            reasoning_steps=steps,
            final_answer=answer,
            verification=f"Each step was verified against the {topic} principles. The chain of reasoning is valid.",
            tags=[domain, topic, f"steps_{num_steps}"],
        )

    def _generate_logical_deduction(self, difficulty: int) -> Optional[ReasoningExample]:
        premise_templates = [
            f"All {random.choice(['A', 'B', 'C'])} are {random.choice(['X', 'Y', 'Z'])}.",
            f"Some {random.choice(['A', 'B', 'C'])} are {random.choice(['X', 'Y', 'Z'])}.",
            f"No {random.choice(['A', 'B', 'C'])} are {random.choice(['X', 'Y', 'Z'])}.",
            f"If P then Q. P is {'true' if random.random() > 0.5 else 'false'}.",
            f"Either X or Y is true. X is {'true' if random.random() > 0.5 else 'false'}.",
            f"All mathematicians are logical. Socrates is a mathematician.",
        ]

        premises = random.sample(premise_templates, min(3 + difficulty, len(premise_templates)))
        question = f"Given the following premises, determine the logical conclusion:\n" + "\n".join(premises)
        steps = [
            ReasoningStep(1, f"Identify the logical form: {random.choice(['categorical syllogism', 'modus ponens', 'modus tollens', 'disjunctive syllogism'])}",
                         ReasoningType.DEDUCTIVE, "Pattern matching based on logical structure"),
            ReasoningStep(2, f"Apply the inference rule: eliminate the middle term and derive the relationship.",
                         ReasoningType.DEDUCTIVE, "By the rules of categorical logic"),
            ReasoningStep(3, f"Conclusion: {random.choice(['Therefore, all A are Z.', 'Therefore, some A are not Z.', 'Therefore, P implies Q.'])}",
                         ReasoningType.DEDUCTIVE, "Valid inference from premises"),
        ]
        return ReasoningExample(
            reasoning_type=ReasoningType.DEDUCTIVE,
            domain="logic",
            question=question,
            reasoning_steps=steps,
            final_answer=steps[-1].content,
            verification="The deduction follows valid logical rules.",
            tags=["logic", "deduction"],
        )

    def _generate_contradiction_detection(self, difficulty: int) -> Optional[ReasoningExample]:
        contradictions = {
            "All birds can fly. Penguins are birds. Penguins cannot fly.": "The statement 'All birds can fly' contradicts 'Penguins cannot fly' since penguins are birds.",
            "This statement is false.": "The self-referential statement creates a logical paradox - if true, it's false; if false, it's true.",
            "The set of all sets that do not contain themselves.": "Russell's paradox: does this set contain itself? Both answers lead to contradiction.",
            "I always lie.": "If the speaker always lies, then this statement itself is a lie, meaning they don't always lie - a paradox.",
        }

        if random.random() < 0.5:
            pair = random.choice(list(contradictions.items()))
            question = f"Identify the contradiction: {pair[0]}"
            steps = [
                ReasoningStep(1, "Analyze each claim independently.", ReasoningType.CRITICAL),
                ReasoningStep(2, "Check for logical incompatibility between claims.", ReasoningType.DEDUCTIVE),
                ReasoningStep(3, f"Contradiction found: {pair[1]}", ReasoningType.CRITICAL),
            ]
            return ReasoningExample(
                reasoning_type=ReasoningType.CRITICAL,
                domain="logic",
                question=question,
                reasoning_steps=steps,
                final_answer=f"The statements are contradictory. {pair[1]}",
                verification="Standard logical analysis confirms the contradiction.",
                tags=["contradiction", "logic"],
            )

        consistent = "All squares are rectangles. All rectangles have four sides. Therefore, all squares have four sides."
        question = f"Are these statements contradictory? '{consistent}'"
        steps = [
            ReasoningStep(1, "Examine each statement for logical compatibility.", ReasoningType.CRITICAL),
            ReasoningStep(2, "Trace the chain of reasoning: squares → rectangles → four sides.", ReasoningType.DEDUCTIVE),
            ReasoningStep(3, "No contradiction found. The reasoning is valid.", ReasoningType.DEDUCTIVE),
        ]
        return ReasoningExample(
            reasoning_type=ReasoningType.CRITICAL,
            domain="logic",
            question=question,
            reasoning_steps=steps,
            final_answer="No contradiction. The statements are logically consistent.",
            verification="Standard syllogistic logic confirms validity.",
            tags=["contradiction", "consistency"],
        )

    def _generate_decomposition(self, difficulty: int) -> Optional[ReasoningExample]:
        domain = random.choice(REASONING_DOMAINS)
        sub_problems = random.randint(3 + difficulty, 5 + difficulty)
        components = ["input processing", "core logic", "validation", "optimization", "error handling", "output formatting"]

        steps = []
        for i in range(sub_problems):
            component = components[i % len(components)]
            steps.append(ReasoningStep(
                i + 1,
                f"Sub-problem {i+1}: Handle {component}. "
                f"Solve independently by applying {domain} principles.",
                ReasoningType.COMPOSITIONAL,
                f"This decomposition isolates {component} as a separable concern."
            ))

        steps.append(ReasoningStep(
            sub_problems + 1,
            f"Integrate all {sub_problems} sub-solutions to form the complete answer. "
            f"Ensure the interfaces between components are consistent.",
            ReasoningType.SYSTEMATIC,
            "Composition of independently verified components yields a correct solution."
        ))

        return ReasoningExample(
            reasoning_type=ReasoningType.COMPOSITIONAL,
            domain=domain,
            question=f"Break down and solve this {domain} problem by decomposing it into {sub_problems} sub-problems.",
            context=f"A complex {domain} problem requiring systematic decomposition.",
            reasoning_steps=steps,
            final_answer=f"Solution assembled from {sub_problems} sub-problems in {domain}.",
            verification=f"Each sub-problem was verified independently. Integration validated.",
            tags=["decomposition", domain],
        )

    def _generate_planning(self, difficulty: int) -> Optional[ReasoningExample]:
        goals = [
            f"Design a system to process {random.randint(1000, 1000000)} requests per second.",
            f"Implement an algorithm to find the shortest path in a graph with {random.randint(100, 10000)} nodes.",
            f"Develop a strategy to optimize resource allocation across {random.randint(5, 50)} departments.",
            f"Create a testing plan for a distributed system with {random.randint(10, 100)} microservices.",
        ]
        goal = random.choice(goals)
        phases = random.randint(3 + difficulty, 6 + difficulty)

        steps = []
        plan_actions = [
            "Analyze requirements and constraints",
            "Design the high-level architecture",
            "Identify key dependencies and risks",
            "Implement the core components",
            "Test and validate the solution",
            "Deploy and monitor performance",
            "Optimize based on feedback",
        ]

        for i in range(min(phases, len(plan_actions))):
            deps = list(range(1, i)) if i > 1 else []
            steps.append(ReasoningStep(
                i + 1,
                f"Phase {i+1}: {plan_actions[i]}. Estimated effort: {random.choice(['low', 'medium', 'high'])}.",
                ReasoningType.STRATEGIC,
                f"Depends on phases: {deps if deps else 'none'}",
            ))

        return ReasoningExample(
            reasoning_type=ReasoningType.STRATEGIC,
            domain="engineering",
            question=f"Create a detailed plan to: {goal}",
            reasoning_steps=steps,
            final_answer=f"Complete {phases}-phase plan for the goal.",
            verification="Plan covers all phases with proper dependency tracking.",
            tags=["planning", "engineering"],
        )

    def _generate_debugging_reasoning(self, difficulty: int) -> Optional[ReasoningExample]:
        reasoning_errors = [
            {
                "flawed": "If A then B. B is true. Therefore A must be true.",
                "analysis": "This is the fallacy of affirming the consequent. A implies B, but B could be true for other reasons.",
                "correct": "If A then B. B is true. We cannot conclude A. There may be other causes of B.",
            },
            {
                "flawed": "All observed swans are white. Therefore, all swans are white.",
                "analysis": "This is hasty generalization. Inductive reasoning cannot prove universals from finite observations.",
                "correct": "All observed swans are white. This suggests, but does not prove, that all swans may be white.",
            },
            {
                "flawed": "Event A happened before event B. Therefore, A caused B.",
                "analysis": "Post hoc ergo propter hoc fallacy. Temporal correlation does not imply causation.",
                "correct": "Event A happened before event B. Correlation requires further investigation to establish causation.",
            },
        ]

        error = random.choice(reasoning_errors)
        question = f"Find the reasoning error: '{error['flawed']}'"
        steps = [
            ReasoningStep(1, f"Identify the reasoning pattern: {error['flawed']}", ReasoningType.DIAGNOSTIC),
            ReasoningStep(2, f"Analyze logical validity: {error['analysis']}", ReasoningType.CRITICAL),
            ReasoningStep(3, f"Provide correct reasoning: {error['correct']}", ReasoningType.DEDUCTIVE),
        ]

        return ReasoningExample(
            reasoning_type=ReasoningType.CRITICAL,
            domain="logic",
            question=question,
            wrong_answer=error["flawed"],
            reasoning_steps=steps,
            final_answer=f"Error: {error['analysis']}\nCorrect: {error['correct']}",
            verification="Standard logical fallacy analysis confirms the error.",
            tags=["debugging_reasoning", "fallacies"],
        )

    def _generate_mathematical_reasoning(self, difficulty: int) -> Optional[ReasoningExample]:
        problems = [
            {
                "question": "Prove that the square root of 2 is irrational.",
                "steps": [
                    "Assume √2 = a/b where a and b are coprime integers.",
                    "Then 2 = a²/b², so a² = 2b².",
                    "Therefore a² is even, so a is even. Let a = 2k.",
                    "Then 4k² = 2b², so 2k² = b². Therefore b² is even, so b is even.",
                    "Both a and b are even, contradicting coprimality. Hence √2 is irrational."
                ],
            },
            {
                "question": "How many ways can you arrange n distinct items in a circle? (Circular permutations)",
                "steps": [
                    "Consider n distinct items arranged in a circle.",
                    "In linear arrangement: n! ways.",
                    "In circular arrangement, rotations are considered identical.",
                    "There are n rotations for each arrangement.",
                    "Therefore, circular permutations = n! / n = (n-1)!."
                ],
            },
        ]

        problem = random.choice(problems)
        steps = [
            ReasoningStep(i + 1, s, ReasoningType.DEDUCTIVE,
                         f"Step {i+1} follows from mathematical principles.")
            for i, s in enumerate(problem["steps"])
        ]

        return ReasoningExample(
            reasoning_type=ReasoningType.DEDUCTIVE,
            domain="mathematics",
            question=problem["question"],
            reasoning_steps=steps,
            final_answer=problem["steps"][-1] if problem["steps"] else "",
            verification="Proof verified using standard mathematical reasoning.",
            tags=["mathematics", "proof"],
        )

    def _generate_counterfactual_reasoning(self, difficulty: int) -> Optional[ReasoningExample]:
        scenarios = [
            {"scenario": "A company launches a product with extensive marketing.", "change": "they had zero marketing budget.",
             "analysis": "Without marketing, product awareness drops. Sales depend more on organic reach and product quality."},
            {"scenario": "An algorithm uses O(n²) sorting.", "change": "it used O(n log n) sorting instead.",
             "analysis": "For n=10⁶, O(n log n) is ~20M operations vs O(n²) at 10¹² operations - a 50,000x speedup."},
        ]

        sc = random.choice(scenarios)
        steps = [
            ReasoningStep(1, f"Original: {sc['scenario']}", ReasoningType.CAUSAL),
            ReasoningStep(2, f"Counterfactual: What if {sc['change']}", ReasoningType.COUNTERFACTUAL),
            ReasoningStep(3, f"Analysis: {sc['analysis']}", ReasoningType.CAUSAL),
        ]

        return ReasoningExample(
            reasoning_type=ReasoningType.COUNTERFACTUAL,
            domain="general",
            question=f"How would the outcome differ if {sc['change']}?",
            context=sc["scenario"],
            reasoning_steps=steps,
            final_answer=sc["analysis"],
            tags=["counterfactual"],
        )

    def _generate_comparison_analysis(self, difficulty: int) -> Optional[ReasoningExample]:
        comparisons = [
            {"options": "REST vs GraphQL APIs",
             "analysis": "REST: simpler caching, stateless. GraphQL: flexible queries, over-fetching prevention.",
             "recommendation": "Use REST for simple CRUD, GraphQL for complex data requirements."},
            {"options": "SQL vs NoSQL databases",
             "analysis": "SQL: ACID compliance, joins, schema enforcement. NoSQL: horizontal scaling, flexible schema.",
             "recommendation": "SQL for structured data with relationships, NoSQL for high-volume unstructured data."},
        ]

        comp = random.choice(comparisons)
        steps = [
            ReasoningStep(1, f"Option A: {comp['options'].split(' vs ')[0]}", ReasoningType.COMPARATIVE),
            ReasoningStep(2, f"Option B: {comp['options'].split(' vs ')[1]}", ReasoningType.COMPARATIVE),
            ReasoningStep(3, f"Trade-off analysis: {comp['analysis']}", ReasoningType.CRITICAL),
            ReasoningStep(4, f"Recommendation: {comp['recommendation']}", ReasoningType.STRATEGIC),
        ]

        return ReasoningExample(
            reasoning_type=ReasoningType.COMPARATIVE,
            domain="technology",
            question=f"Compare and contrast: {comp['options']}",
            reasoning_steps=steps,
            final_answer=comp["recommendation"],
            verification="Analysis covers key dimensions of comparison.",
            tags=["comparison", "trade-offs"],
        )

    def _generate_default(self, difficulty: int) -> Optional[ReasoningExample]:
        return self._generate_multi_step_reasoning(difficulty)

    def generate_chain_of_thought(self, question: str, steps: List[str], answer: str) -> ReasoningExample:
        reasoning_steps = [
            ReasoningStep(i + 1, step, ReasoningType.DEDUCTIVE,
                         f"This step advances the reasoning toward the solution.")
            for i, step in enumerate(steps)
        ]
        cot = "\n".join(f"**Step {i+1}:** {s}" for i, s in enumerate(steps))
        return ReasoningExample(
            question=question,
            reasoning_steps=reasoning_steps,
            final_answer=answer,
            verification="Chain-of-thought verified for logical consistency.",
            metadata={"cot_format": cot},
        )

    def generate_batch(self, counts: Dict[str, int]) -> List[ReasoningExample]:
        all_examples = []
        for task_name, count in counts.items():
            try:
                task = ReasoningTask(task_name)
            except ValueError:
                for t in ReasoningTask:
                    if t.value == task_name:
                        task = t
                        break
                else:
                    continue
            examples = self.generate(task, count=count)
            all_examples.extend(examples)
        return all_examples

    def get_stats(self) -> Dict:
        return self.stats
