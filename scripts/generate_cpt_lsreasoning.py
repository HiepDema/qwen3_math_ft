"""Generate 100 CPT samples for LSReasoning math domain.

Plain text passages about math reasoning - no instruction/response format,
just natural text to help model absorb math vocabulary and patterns.

Usage:
    python scripts/generate_cpt_lsreasoning.py
"""

import json
import random
from pathlib import Path


def generate_cpt_data(num_samples=100, seed=42):
    random.seed(seed)
    passages = []

    # Arithmetic theory
    arithmetic = [
        "Addition is the process of combining two or more numbers to find their total. For example, 25 + 17 = 42. When adding, we combine the ones place first, then the tens place, carrying over when the sum exceeds 9.",
        "Subtraction finds the difference between two numbers. To compute 84 - 37, we subtract the ones place (4-7 requires borrowing), giving 84 - 37 = 47. Subtraction is the inverse of addition.",
        "Multiplication is repeated addition. For instance, 6 × 8 = 48 means adding 6 eight times. The multiplication table is fundamental to arithmetic fluency.",
        "Division splits a number into equal parts. 144 ÷ 12 = 12. Division is the inverse of multiplication. When dividing, we ask: how many times does the divisor fit into the dividend?",
        "The order of operations (PEMDAS) states: Parentheses first, then Exponents, then Multiplication and Division (left to right), then Addition and Subtraction (left to right).",
        "When adding negative numbers, -5 + (-3) = -8. Adding two negatives gives a more negative result. But -5 + 8 = 3, because the positive number has greater magnitude.",
        "Subtracting a negative is the same as adding: 7 - (-3) = 7 + 3 = 10. Two negatives make a positive in subtraction.",
        "Multiplying two negative numbers gives a positive: (-4) × (-6) = 24. A negative times a positive gives a negative: (-4) × 6 = -24.",
    ]

    # Linear equations
    linear_eq = [
        "A linear equation has the form ax + b = c, where x is the unknown. To solve, isolate x: subtract b from both sides to get ax = c - b, then divide by a to get x = (c-b)/a.",
        "Solving 3x + 7 = 22: subtract 7 from both sides gives 3x = 15, then divide by 3 gives x = 5. We can verify: 3(5) + 7 = 15 + 7 = 22. Correct.",
        "Two-step equations require two operations to isolate x. For 2x - 5 = 11: first add 5 to get 2x = 16, then divide by 2 to get x = 8.",
        "When solving equations, whatever operation we perform on one side must be done to the other side to maintain equality. This is the balance principle.",
        "For the equation 5x + 12 = 47, subtract 12: 5x = 35, divide by 5: x = 7. Check: 5(7) + 12 = 35 + 12 = 47.",
        "Equations with negative coefficients: -3x + 9 = 0. Subtract 9: -3x = -9. Divide by -3: x = 3. Dividing by a negative flips no signs here since both sides are negative.",
        "The equation 4x - 20 = 0 has solution x = 5. Setting an expression equal to zero and solving is a common pattern in algebra.",
        "Multi-step: 2(x + 3) = 14. First distribute: 2x + 6 = 14. Then subtract 6: 2x = 8. Finally divide by 2: x = 4.",
    ]

    # Fractions
    fractions = [
        "A fraction represents a part of a whole. The fraction 3/4 means 3 parts out of 4 equal parts. The top number is the numerator, the bottom is the denominator.",
        "To simplify a fraction, divide both numerator and denominator by their greatest common divisor (GCD). For 12/18: GCD(12,18) = 6, so 12/18 = 2/3.",
        "Adding fractions requires a common denominator: 1/3 + 1/4 = 4/12 + 3/12 = 7/12. Find the LCD (least common denominator) first.",
        "Multiplying fractions: multiply numerators together and denominators together. 2/3 × 4/5 = 8/15. No common denominator needed.",
        "Dividing fractions: multiply by the reciprocal. 3/4 ÷ 2/5 = 3/4 × 5/2 = 15/8. Flip the second fraction and multiply.",
        "A negative fraction like -6/3 simplifies to -2. Divide the absolute values and apply the negative sign: 6 ÷ 3 = 2, so -6/3 = -2.",
        "To convert a fraction to a decimal, divide numerator by denominator: 3/8 = 0.375. Some fractions produce repeating decimals: 1/3 = 0.333...",
        "Improper fractions have a numerator larger than the denominator: 7/4 = 1 and 3/4 as a mixed number. To convert: divide 7 by 4 = 1 remainder 3.",
    ]

    # Word problems
    word_problems = [
        "Word problems translate real situations into equations. 'If each item costs $4 and total is $84, how many items?' becomes: 4x = 84, so x = 21 items.",
        "Rate problems: If a car travels 60 mph for 3 hours, distance = rate × time = 60 × 3 = 180 miles. The formula d = rt is fundamental.",
        "Age problems: 'Tom is 3 years older than Sam. Together their ages sum to 25.' Let Sam = x, Tom = x+3. Then x + (x+3) = 25, so 2x = 22, x = 11. Sam is 11, Tom is 14.",
        "Percentage problems: 'What is 15% of 80?' Convert: 0.15 × 80 = 12. Percentages are fractions with denominator 100.",
        "Profit problems: Cost = $50, selling price = $65. Profit = selling price - cost = $65 - $50 = $15. Profit margin = 15/50 = 30%.",
        "Mixture problems combine different quantities. Mixing 3 liters of 20% solution with 2 liters of 50% solution: total solute = 0.6 + 1.0 = 1.6 liters in 5 liters = 32%.",
        "Work problems: If A finishes in 6 hours and B in 4 hours, together they complete 1/6 + 1/4 = 2/12 + 3/12 = 5/12 per hour. Time together = 12/5 = 2.4 hours.",
        "Consecutive integer problems: Three consecutive integers sum to 45. Let them be x, x+1, x+2. Then 3x + 3 = 45, 3x = 42, x = 14. The integers are 14, 15, 16.",
    ]

    # Solving strategies
    strategies = [
        "When solving equations, always check your answer by substituting back into the original equation. If both sides are equal, the solution is correct.",
        "Estimation helps verify answers. If solving 49 × 21, estimate: 50 × 20 = 1000. Exact answer: 1029. Close to estimate, so likely correct.",
        "Breaking complex problems into steps makes them manageable. Identify what's given, what's asked, set up the equation, solve, and verify.",
        "Common mistakes in algebra: forgetting to distribute negative signs, dividing only one term when the whole side needs division, and sign errors with negatives.",
        "Mental math shortcuts: to multiply by 9, multiply by 10 and subtract once. 9 × 7 = 70 - 7 = 63. To multiply by 5, divide by 2 and multiply by 10.",
        "When a problem asks 'how many more', it means find the difference. 'How many more' = larger - smaller. This is a subtraction problem.",
        "Unit analysis helps word problems: if price is dollars/item and you want total dollars, multiply price × items. Units: ($/item)(items) = $.",
        "Inverse operations undo each other: addition undoes subtraction, multiplication undoes division. Use inverse operations to isolate variables.",
    ]

    all_texts = arithmetic + linear_eq + fractions + word_problems + strategies
    random.shuffle(all_texts)

    # Generate additional computed examples
    for _ in range(num_samples - len(all_texts)):
        choice = random.choice(["add", "sub", "mul", "div", "eq", "frac"])

        if choice == "add":
            a, b = random.randint(-50, 50), random.randint(-50, 50)
            passages.append({"text": f"Computing {a} + {b}: the sum is {a+b}. {'Both positive, result positive.' if a > 0 and b > 0 else 'Mixed signs, take the sign of the larger magnitude.'}"})
        elif choice == "sub":
            a, b = random.randint(-50, 50), random.randint(-50, 50)
            passages.append({"text": f"Subtracting: {a} - {b} = {a} + ({-b}) = {a-b}. Subtracting is adding the opposite."})
        elif choice == "mul":
            a, b = random.randint(-12, 12), random.randint(-12, 12)
            if a == 0 or b == 0:
                a, b = 3, 7
            passages.append({"text": f"Multiplication: {a} × {b} = {a*b}. {'Positive result (same signs).' if (a>0)==(b>0) else 'Negative result (different signs).'}"})
        elif choice == "div":
            b = random.choice([i for i in range(-10, 11) if i != 0])
            result = random.randint(-10, 10)
            a = b * result
            passages.append({"text": f"Division: {a} ÷ {b} = {result}. We verify: {b} × {result} = {a}. Correct."})
        elif choice == "eq":
            coeff = random.choice([i for i in range(2, 8)])
            x_val = random.randint(-10, 10)
            const = random.randint(-20, 20)
            rhs = coeff * x_val + const
            passages.append({"text": f"Solving {coeff}x + {const} = {rhs}: subtract {const} from both sides gives {coeff}x = {rhs - const}. Divide by {coeff}: x = {x_val}."})
        else:
            num = random.randint(-20, 20)
            den = random.choice([i for i in range(2, 10)])
            from math import gcd
            g = gcd(abs(num), den)
            sn, sd = num // g, den // g
            passages.append({"text": f"Simplifying the fraction {num}/{den}: GCD({abs(num)},{den}) = {g}. So {num}/{den} = {sn}/{sd}."})

    for t in all_texts[:num_samples - len(passages)]:
        passages.append({"text": t})

    random.shuffle(passages)
    return passages[:num_samples]


def main():
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating 100 CPT samples for math reasoning...")
    passages = generate_cpt_data(100, seed=42)

    cpt_path = output_dir / "cpt_lsreasoning.jsonl"
    with open(cpt_path, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Saved {len(passages)} samples to {cpt_path}")
    print(f"\nSamples:")
    for p in passages[:5]:
        print(f"  {p['text'][:100]}...")


if __name__ == "__main__":
    main()
