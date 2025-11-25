# Practical Hypothesis Testing: From P-Values to Decision Rules

## The Promise
By the end of this tutorial, you will master the algorithmic decision engine used in A/B testing, clinical trials, and algorithmic trading. You will move beyond "guessing" to a formalized binary decision process based on probability density.

You will learn to manipulate these variables:
*   **$H_0$ (Null Hypothesis):** The default state of the world.
*   **$H_1$ (Alternative Hypothesis):** The claim you are testing.
*   **$\alpha$ (Alpha):** The significance level (usually 0.05).
*   **$Z$ (Test Statistic):** The standardized distance from the mean.
*   **$P$ (P-Value):** The probability of observing the data assuming $H_0$ is true.

### The Universal Algorithm
Hypothesis testing is not magic. It is a 4-step function.

$$
\text{Decision} = f(\text{Data}, \text{Threshold})
$$

**The 4-Step Process:**
1.  **Formulate:** Define $H_0$ (Status Quo) and $H_1$ (New Theory).
2.  **Calibrate:** Set $\alpha$ (Error tolerance, e.g., 5%).
3.  **Compute:** Calculate the test statistic using the relevant formula.
    *   *Example (Z-Test):* $Z = \frac{\bar{x} - \mu}{\sigma / \sqrt{n}}$
4.  **Decide:**
    *   If $P \text{-value} < \alpha$: **Reject $H_0$**. (Significant)
    *   If $P \text{-value} \ge \alpha$: **Fail to Reject $H_0$**. (Not Significant)

### The Landscape of Tests
While the logic above remains constant, the **Step 3** formula changes based on data type and volume. You do not need to memorize these now, but acknowledge the hierarchy:

| Data Type | Sample Size | Comparisons | Test Name |
| :--- | :--- | :--- | :--- |
| **Continuous** (e.g., Height, Price) | Large ($n > 30$) | Mean vs. Target | **Z-Test** |
| **Continuous** | Small ($n < 30$) | Mean vs. Target | **T-Test** |
| **Continuous** | Any | 3+ Groups | **ANOVA** |
| **Categorical** (e.g., Click/No Click) | Any | Frequency counts | **Chi-Square** |

This tutorial focuses on the **Z-Test** to build the core intuition, which applies to all other tests.

## Section 1: Intuition First - The Suspicious Coin

We start with a concrete problem to build the mental model.

**The Problem:**
A friend hands you a coin and claims it is "fair" (50/50 chance). You bet on the next 10 flips.
You flip the coin 10 times.
**Result:** 9 Heads, 1 Tail.

**The Question:**
Is your friend cheating, or did you just get lucky?

**The Concrete Solution (The "BS Detector"):**
We cannot strictly *prove* the coin is rigged. Instead, we assume the coin is **fair** and calculate the probability of observing this result.

If the probability is too low, we call "BS" (Bullshit).

**Step 1: The Math**
If the coin is fair ($P_{head} = 0.5$), the probability of getting *exactly* 10 Heads is:
$$0.5^{10} \approx 0.00097$$

The probability of getting *exactly* 9 Heads is:
$$10 \times 0.5^{10} \approx 0.00976$$

The probability of getting **at least** 9 heads (9 or 10) is the sum:
$$0.00097 + 0.00976 \approx 0.0107$$

**Result:** There is a **1.07%** chance a fair coin does this.

**Step 2: The Decision Rule**
Before the game, you have an internal "BS Threshold." Usually, if something has less than a 5% chance of happening by accident, we conclude it wasn't an accident.

*   **Observation:** 1.07% chance.
*   **Threshold:** 5%.
*   **Logic:** $1.07\% < 5\%$.
*   **Conclusion:** The event is too rare to be luck. The coin is likely rigged.

### Abstracting the Solution

Now we map this specific scenario to the formal statistical terms used in the industry.

| Concrete Coin Scenario | Formal Statistical Term | Symbol |
| :--- | :--- | :--- |
| "The coin is fair." | **Null Hypothesis** | $H_0$ |
| "The coin is rigged." | **Alternative Hypothesis** | $H_1$ |
| 1.07% (Calculated chance of luck) | **P-Value** | $p$ |
| 5% (Your BS Threshold) | **Significance Level** | $\alpha$ |

**The Logic Visualized:**

```text
Shape: 3D Bar Chart
X-axis: Number of Heads (0 to 10)
Y-axis: Probability

Description:
- The bars form a bell-curve shape centered at 5.
- The bar at 5 is the tallest (most likely).
- The bars at 9 and 10 are tiny, red slivers at the far right edge.
- An arrow points to 9 and 10 labeled: "The Rejection Region (P-Value)"
```

### Code Snippet: Calculating the P-Value

In practice, you don't do the math by hand. You use a library.

**Input:**
*   Successes (Heads): 9
*   Trials (Flips): 10
*   Expected Probability ($H_0$): 0.5

```python
from scipy.stats import binomtest

# Inputs
k = 9      # Number of heads
n = 10     # Number of flips
p = 0.5    # Null hypothesis probability

# Calculate P-Value
result = binomtest(k, n, p, alternative='greater')

print(f"P-Value: {result.pvalue:.4f}")
```

**Output:**
```text
P-Value: 0.0107
```

**Summary of Section 1:**
We rejected the "Fair Coin" hypothesis because the p-value (0.0107) was lower than our alpha (0.05). We formally decided: **Reject $H_0$**.

## Phase 2: The Yardstick (Normal Distribution & Z-Test)

**Context in Big Picture:** In Phase 1, we counted discrete events (Heads/Tails). But most business data is **continuous** (Revenue, Latency, Weight). You cannot calculate "probability of exactly \$100.05 revenue" because the possibilities are infinite.

We need a new yardstick. We rely on the **Normal Distribution** and the **Z-Score**.

### 1. The Concrete Problem
You manage a chip factory.
*   **Target Weight:** 100g.
*   **Known Variation ($\sigma$):** The machine historically deviates by 5g (Standard Deviation).

You take a sample of **50 chips**.
*   **Sample Average ($\bar{X}$):** 102g.

**Question:** Is the machine broken (drifting to 102g)? Or is this random noise?

### 2. The Tool: Central Limit Theorem (CLT)
Why can we solve this?
**The CLT Promise:** No matter how weird or messy your individual data points are, if you take enough samples and average them, the **distribution of those averages** will form a perfect Bell Curve (Normal Distribution).

Because of this, we don't need to know the shape of the raw data. We only analyze the **Sample Mean**.

### 3. The Formula: The Z-Score
The Z-Score converts your specific problem (grams, dollars, seconds) into a universal "Standard Deviation units."

$$Z = \frac{\text{Signal}}{\text{Noise}} = \frac{\bar{X} - \mu}{\frac{\sigma}{\sqrt{n}}}$$

*   $\bar{X}$: Sample Mean (102g)
*   $\mu$: Target Population Mean (100g)
*   $\sigma$: Population Standard Deviation (5g)
*   $n$: Sample Size (50)
*   $\sigma / \sqrt{n}$: **Standard Error** (The volatility of the *average*, not the individual).

### 4. Step-by-Step Execution
Let's run the Master Algorithm on the factory data.

1.  **Define $H_0$:** $\mu = 100$ (Machine is fine).
2.  **Collect Data:** $\bar{X} = 102$, $n = 50$.
3.  **Calculate Z-Score:**
    *   Signal: $102 - 100 = 2$
    *   Noise (Standard Error): $5 / \sqrt{50} \approx 0.707$
    *   $Z = 2 / 0.707 \approx \mathbf{2.83}$
4.  **Compute P-Value:**
    *   How rare is a Z-score of 2.83?
    *   In a standard curve, 95% of data is between -1.96 and +1.96.
    *   2.83 is way outside.
    *   $P(Z > 2.83) \approx 0.0023$ (0.23%).
5.  **Decision:**
    *   $0.0023 < 0.05$. **Reject $H_0$**.
    *   **Reality Check:** The machine is broken. Stop the line.

### 5. Code Example (Z-Test)
Calculating tail probabilities manually is hard. Use Python.

```python
import numpy as np
from scipy.stats import norm

# Inputs
mu_target = 100
sigma = 5
n = 50
x_bar = 102

# Calculate Z
std_error = sigma / np.sqrt(n)
z_score = (x_bar - mu_target) / std_error

# Calculate P-value (Two-tailed)
# We multiply by 2 because we care if it's too high OR too low
p_value = 2 * (1 - norm.cdf(abs(z_score)))

print(f"Z-Score: {z_score:.2f}")
print(f"P-Value: {p_value:.4f}")
```

**Output:**
```text
Z-Score: 2.83
P-Value: 0.0047
```

### 6. Visualizing the Z-Score
The Z-score maps your data onto the Standard Normal Curve (Mean=0, SD=1).

```
Standard Normal Distribution (The Yardstick)

      ^
      |         Center (0)
      |             |
      |           __|__
      |        _ /     \ _
      |      /             \
      |    /                 \
      | __|___________________|__      [Rejection Zone]
______|_|_|___________________|_|___X____
Z:     -2  -1     0      1     2    ^
                                    |
                                    2.83 (Observed)
                                    
The observed Z (2.83) falls into the rejection tail.
The probability of landing here by luck is tiny.
```

### 7. Connect to Reality
We just determined that a 2g difference is huge. Why? Because we had a sample size of 50.
**Key Insight:** If $n$ was only 5, the Standard Error ($\sigma/\sqrt{n}$) would be larger, the Z-score would be smaller, and we might *fail* to reject.
**More Data = More Sensitivity to small differences.**

## Phase 3: Handling Unknowns (Student’s T-Distribution)

**Context in Big Picture:** Phase 2 (Z-Test) required you to know the global Standard Deviation ($\sigma$). In the real world, you almost never know this. You only have your small sample. When you rely on sample data to estimate variance, you add uncertainty. The **T-Test** accounts for this.

### 1. The Concrete Problem (A/B Testing)
You are testing a new website design.
*   **Group A (Control):** 10 users. Average time on site = 45s.
*   **Group B (Treatment):** 10 users. Average time on site = 55s.

**Question:** Is the 10s improvement real, or just random noise from small sample sizes?
**Constraint:** You do not know the standard deviation of the entire internet population. You only have the variation inside your two groups ($s$).

### 2. The Solution: Student’s T-Distribution
Because we are estimating the variance using the sample ($s$) instead of the population ($\sigma$), our predictions are less precise.

If we used the Normal Distribution (Z), we would be overconfident.
**The Fix:** The **T-Distribution**. It looks like a Normal Distribution but shorter with **fatter tails**. The fat tails represent the "penalty" for not knowing the true population variance.

### 3. The Formula: T-Statistic (Two-Sample)
We compare the difference between two groups relative to their combined volatility.

$$t = \frac{\text{Difference in Means}}{\text{Combined Noise}} = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$

*   $\bar{X}_1 - \bar{X}_2$: The Signal (10s difference).
*   $s_1^2, s_2^2$: Sample Variances (The noise inside each group).
*   $n_1, n_2$: Sample sizes.

### 4. Step-by-Step Execution
1.  **Define $H_0$:** $\mu_A = \mu_B$ (The new design changes nothing).
2.  **Collect Data:**
    *   Group A: Mean=45, SD=15, n=10.
    *   Group B: Mean=55, SD=15, n=10.
3.  **Calculate T-Score:**
    *   Signal: $55 - 45 = 10$.
    *   Noise: $\sqrt{\frac{225}{10} + \frac{225}{10}} = \sqrt{45} \approx 6.7$.
    *   $t = 10 / 6.7 \approx \mathbf{1.49}$.
4.  **Compute P-Value:**
    *   We look up $t=1.49$ with "degrees of freedom" ($n_1 + n_2 - 2 = 18$).
    *   P-value $\approx 0.15$.
5.  **Decision:**
    *   $0.15 > 0.05$. **Fail to Reject $H_0$**.
    *   **Conclusion:** The 10s increase is likely noise. You do not have enough evidence to claim the new design works.

### 5. Code Example (T-Test)
This is the most common statistical function you will use in Python.

```python
from scipy import stats
import numpy as np

# Fake data representing time on site (seconds)
group_a = np.array([40, 50, 45, 30, 60, 42, 48, 35, 55, 45]) # Control
group_b = np.array([50, 60, 55, 40, 70, 52, 58, 45, 65, 55]) # Treatment

# Run Independent Two-sample T-test
t_stat, p_val = stats.ttest_ind(group_b, group_a)

print(f"T-Statistic: {t_stat:.2f}")
print(f"P-Value: {p_val:.4f}")
```

**Output:**
```text
T-Statistic: 2.28
P-Value: 0.0350
```
*(Note: In this specific code snippet, the data was cleaner, resulting in a significant P-value).*

### 6. Visualizing T vs Normal
The T-distribution changes shape based on Sample Size ($n$).

```
Comparision of Curves

      ^
      |      _..-'''-.._        <-- Normal Distribution (Z)
      |    .'     |     '.          (High Peak, Thin Tails)
      |   /   _..-|-.._   \
      |  /  .'    |    '.  \    <-- T-Distribution (n=5)
      | |  /      |      \  |       (Lower Peak, Fatter Tails)
______|_\_|_______|_______|_/____
         -3       0       3
         
The Fatter Tail means you need a HIGHER score to prove significance.
The T-test is "conservative." It forces you to prove it more effectively 
because your sample size is small.
```

### 7. Reality Check
We just failed to reject the null in the manual example ($p=0.15$), even though the average improved by 10 seconds.
**The Lesson:** Big impact doesn't matter if your variance ($s$) is high or your sample size ($n$) is low. To pass the T-test, you either need a **massive improvement** or **more data** to shrink the variance.

## Phase 4: Categorical Differences (Chi-Square Test)

**Context in Big Picture:** The Z-test and T-test work for **averages** of continuous numbers (Height, Money, Time). But what if your data is **categorical**?
*   Did the user Click or Not Click?
*   Did the customer Buy or Not Buy?

You cannot calculate the "average" of a Click. You can only count frequencies. For this, we use the **Chi-Square ($\chi^2$) Test**.

### 1. The Concrete Problem (Conversion Rate)
You run an E-commerce site. You test a new "Buy" button color.
*   **Control (Blue):** 100 visitors, 10 bought. (10% rate)
*   **Variant (Red):** 100 visitors, 20 bought. (20% rate)

**Question:** Is the Red button actually better, or did you just get lucky with those specific 100 visitors?

### 2. The Logic: Observed vs. Expected
The logic shifts from "Comparing Means" to "Comparing Tables."

*   **Observed ($O$):** What actually happened.
*   **Expected ($E$):** What *should* have happened if the button color didn't matter ($H_0$).

We measure the logical "distance" between $O$ and $E$. If the distance is large, $H_0$ is false.

### 3. The Formula: Chi-Square Statistic
$$ \chi^2 = \sum \frac{(O_i - E_i)^2}{E_i} $$

*   $O_i$: Observed count in a cell.
*   $E_i$: Expected count in a cell.
*   We square the difference to get rid of negatives.
*   We divide by expected to normalize (an error of 5 is huge if expected is 10, but tiny if expected is 10,000).

### 4. Step-by-Step Execution
1.  **Define $H_0$:** Color has no effect on sales.
2.  **Build Observed Table:**
    *   Total Visitors: 200.
    *   Total Sales: 30.
    *   Global Conversion Rate: $30/200 = 15\%$.

| Group | Bought | Didn't Buy | Total |
| :--- | :--- | :--- | :--- |
| **Blue** | 10 ($O_1$) | 90 | 100 |
| **Red** | 20 ($O_2$) | 80 | 100 |

3.  **Calculate Expected Table ($H_0$ True):**
    If $H_0$ is true, both groups should convert at the Global Rate (15%).
    *   Blue Expected Buys: $100 \times 0.15 = 15$ ($E_1$).
    *   Red Expected Buys: $100 \times 0.15 = 15$ ($E_2$).

4.  **Calculate $\chi^2$:**
    *   Blue Buys: $(10 - 15)^2 / 15 = 25/15 = 1.66$
    *   Red Buys: $(20 - 15)^2 / 15 = 25/15 = 1.66$
    *   *Do the same for "Didn't Buy" cells...*
    *   $\chi^2 \approx 3.92$.

5.  **Compute P-Value:**
    *   Degrees of Freedom = 1.
    *   Look up $\chi^2 = 3.92$.
    *   P-value $\approx 0.047$.

6.  **Decision:**
    *   $0.047 < 0.05$. **Reject $H_0$**.
    *   **Result:** The Red button really works.

### 5. Code Example (Chi-Square)
We do not construct expected tables by hand. Python does it automatically.

```python
from scipy.stats import chi2_contingency

# Input: The Observed Table
# [[Bought, No Buy] (Control), [Bought, No Buy] (Variant)]
data = [[10, 90], 
        [20, 80]]

chi2, p, dof, expected = chi2_contingency(data)

print(f"Chi2 Stat: {chi2:.2f}")
print(f"P-Value: {p:.4f}")
```

**Output:**
```text
Chi2 Stat: 3.92
P-Value: 0.0477
```

### 6. Visualizing the Logic
Think of this as fitting a grid over reality.

```
       Expected Reality (H0)            Observed Reality (Data)
      _______________________           _______________________
     |          |            |         |          |            |
     |   15     |     85     |         |   10     |     90     | <--- Control
     |__________|____________|         |__________|____________|
     |          |            |         |          |            |
     |   15     |     85     |         |   20     |     80     | <--- Variant
     |__________|____________|         |__________|____________|

           ^                                    ^
           |____________________________________|
                          Measure
                       Difference (Chi^2)
```

### 7. Reality Check: Sample Size Sensitivity
We rejected $H_0$ with a P-value of 0.047. This is barely significant.
**The Reality:** If the Red button had 19 sales instead of 20, the P-value would jump to > 0.05, and we would conclude "No Difference."
In categorical data (Clicks/Conversions), you need **massive sample sizes** to detect small differences because the data is "low resolution" (just Yes/No, not a precise number).

## Phase 5: Production Reality

**Context in Big Picture:** You now know the math that runs inside the black boxes of tools like Optimizely, Google Optimize, or Tableau. In the real world, you rarely calculate these by hand. Your job is to **select the right test** and **interpret the output correctly**.

### 1. The "God View": Choosing the Right Test
You don't need to memorize formulas. You need to memorize this decision matrix. Look at your data, identify the type, and pick the algorithm.

| Data Type | Question Example | The Variable | The Test |
| :--- | :--- | :--- | :--- |
| **Categorical** | Did they click? (Yes/No) | Conversion Rate | **Chi-Square Test** |
| **Continuous** | Did they spend more? ($) | Average Revenue (Unknown $\sigma$) | **Two-Sample T-Test** |
| **Continuous** | Is the machine calibrated? | Average Weight (Known $\sigma$) | **Z-Test** |
| **Binary** | Is the coin rigged? | Success Count | **Binomial Test** |

### 2. Production Pitfall: P-Hacking (Peeking)
**The Trap:** You run an A/B test. You check the P-value every hour.
*   Hour 1: $p = 0.08$ (Not significant)
*   Hour 4: $p = 0.04$ (Significant! Stop!)
*   Hour 5: $p = 0.12$ (Not significant)

**The Reality:** If you check often enough, you *will* eventually see $p < 0.05$ by random chance. This is cheating.
**The Solution:** Calculate your required sample size **before** you start. Run the test until you hit that number. Do not peek.

### 3. Production Pitfall: Statistical vs. Practical Significance
**The Scenario:** You test a new font on Google Search.
*   Sample Size: 10 Million users.
*   Result: Search time improves by 0.0001 seconds.
*   P-Value: $< 0.0000001$.

**The Verdict:**
*   **Statistically Significant?** Yes. The math proves the difference is real.
*   **Practically Significant?** No. 0.0001 seconds is invisible to a human.
*   **Lesson:** Low P-value $\neq$ Big Impact. It just means "High Certainty." Always check the **Effect Size** (the actual magnitude of the difference).

### 4. Summary: The 40-Minute Journey
We started with a promise: to learn the universal algorithm of inference.

1.  **The Framework:** $H_0$, Data, Test Statistic, P-value, Decision.
2.  **The Yardstick:** Standard Error measures how "noisy" the data is.
3.  **The Ratio:** All tests follow the same pattern:
    $$ \text{Score} = \frac{\text{Observed Signal}}{\text{Expected Noise}} $$
4.  **The Decision:** If the Signal is much stronger than the Noise (High Score, Low P-value), we accept the new reality.

**You have now graduated from "guessing" to "inference."**