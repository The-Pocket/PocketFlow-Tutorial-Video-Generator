# Give me 1 hour, I will make differential calculus click forever

---

## Introduction

You've heard the terms: calculus, derivatives, differentiation. They are not just abstract math topics; they are the fundamental language for describing change. Understanding them is a non-negotiable requirement for every quantitative field: physics, engineering, computer science, economics, and even deep learning.

This tutorial is designed to build your intuition and then formalize it. No hand-waving, no magic.

This table is the core of differential calculus. It might look like a list of abstract rules to memorize. It is not.

**The Toolbox You Will Master**

| Category                | Function `f(x)`       | Derivative `f'(x)`                |
| ----------------------- | --------------------- | --------------------------------- |
| **Basic Rules**         | `c` (constant)        | `0`                               |
|                         | `xⁿ`                  | `nxⁿ⁻¹`                           |
| **Essential Functions** | `eˣ`                  | `eˣ`                              |
|                         | `ln(x)`               | `1/x`                             |
| **Trigonometric**       | `sin(x)`              | `cos(x)`                          |
|                         | `cos(x)`              | `-sin(x)`                         |
|                         | `tan(x)`              | `sec²(x)`                         |
|                         | `sec(x)`              | `sec(x)tan(x)`                    |
|                         | `csc(x)`              | `-csc(x)cot(x)`                   |
|                         | `cot(x)`              | `-cot²(x)`                        |
| **Inverse Trig**        | `arcsin(x)` or `sin⁻¹(x)` | `1 / √(1 - x²)`                   |
|                         | `arccos(x)` or `cos⁻¹(x)` | `-1 / √(1 - x²)`                  |
|                         | `arctan(x)` or `tan⁻¹(x)` | `1 / (1 + x²)`                    |
| **General Exponentials**| `aˣ`                  | `aˣ ln(a)`                        |
| **& Logarithms**        | `logₐ(x)`             | `1 / (x ln(a))`                   |
| **Combination Rules**   | `f(x) ± g(x)`         | `f'(x) ± g'(x)`                   |
|                         | `f(x)g(x)`            | `f'(x)g(x) + f(x)g'(x)`           |
|                         | `f(x) / g(x)`         | `[f'g - fg'] / g²`                |
|                         | `f(g(x))`             | `f'(g(x)) * g'(x)`                |

**The Promise:** In the next hour, we will deconstruct this entire table. You will not just learn these rules; you will build them from the ground up. You will understand precisely *why* the derivative of `x²` is `2x`, and how the more complex rules are logical extensions of one simple idea.

By the end, you will look at any function on this list and know its derivative, not because you memorized it, but because you understand the mechanics behind it.

To achieve this, our approach will be grounded in a single, practical problem that we will follow from start to finish. Let's begin.

## **Section 1: The Foundation - Understanding Limits**

#### The Big Picture
Before we can calculate an *instantaneous* rate of change, we need a formal way to talk about getting infinitely close to a point without actually being at that point. This concept is the **limit**. It is the microscope of calculus, allowing us to zoom in on a function's behavior at a specific, single instant.

#### The Motivating Problem
Let's return to our car, whose position is given by `p(t) = t²`. We want to find its speed at the exact moment `t=2`.

We can't measure speed at a single instant directly. We can only calculate *average* speed over an interval of time using this formula:

**Average Speed = (Change in Position) / (Change in Time) = Δp / Δt**

For example, the average speed between `t=2` and `t=3` is:
*   `Δp = p(3) - p(2) = 3² - 2² = 5` meters
*   `Δt = 3 - 2 = 1` second
*   **Average Speed = 5 / 1 = 5 m/s**

But this is a crude average over a whole second. To get closer to the *instantaneous* speed at `t=2`, we must shrink the time interval.

#### Building Intuition: The Squeeze
Let's calculate the average speed over progressively smaller intervals, each starting at `t=2`.

| Time Interval `[t₁, t₂]` | Change in Time `Δt` | Change in Position `Δp = t₂² - 2²` | Average Speed `Δp / Δt` |
| ------------------------- | ------------------- | -------------------------------------- | --------------------------- |
| `[2, 3]`                  | 1.0                 | `9 - 4 = 5`                            | 5.0 m/s                     |
| `[2, 2.1]`                | 0.1                 | `4.41 - 4 = 0.41`                      | 4.1 m/s                     |
| `[2, 2.01]`               | 0.01                | `4.0401 - 4 = 0.0401`                  | 4.01 m/s                    |
| `[2, 2.001]`              | 0.001               | `4.004001 - 4 = 0.004001`              | 4.001 m/s                   |
| `[2, 2.000001]`           | 0.000001            | `...`                                  | 4.000001 m/s                |

The pattern is undeniable. As our time interval `Δt` gets closer and closer to zero, the average speed gets closer and closer to **4 m/s**. We say that **the limit of the average speed as Δt approaches 0 is 4.**

#### Geometric Intuition
We can visualize this on a graph.

*   **Function to Plot:** `y = x²`. This is a simple parabola. The x-axis is time `t`, and the y-axis is position `p`.
*   **Point of Interest:** The point `(2, 4)` on the parabola, representing the car's position at `t=2`.

1.  The average speed between two points, like `(2, 4)` and `(3, 9)`, is the slope of the straight line connecting them. This is called a **secant line**.
2.  As we shrink our time interval, we are sliding the second point (e.g., `(2.1, 4.41)`) down the curve toward our point of interest, `(2, 4)`.
3.  The secant line pivots as the second point moves closer.

```
Start with a secant line between t=2 and t=3 
Bring t=3 closer to t=2, e.g., t=2.1
The secant line pivots
Bring the point even closer, e.g., t=2.01
The secant line pivots more, becoming a better approximation of the slope *at* t=2
In the limit, the secant line becomes the **tangent line** at t=2
```

The **instantaneous speed** is the slope of this final **tangent line**—the line that just skims the curve at that single point. The limit is the tool that lets us find the slope of that line.

#### Formalization: Limit Notation
We write the concept of a limit with the following notation:
`lim (x→c) f(x) = L`

This statement is read as: "The limit of the function `f(x)` as `x` approaches `c` is `L`."

| Component           | Meaning                                                            | In our Example                                                              |
| ------------------- | ------------------------------------------------------------------ | --------------------------------------------------------------------------- |
| `lim`               | The limit operator. Signals we are performing this "approaching" process. | `lim`                                                                       |
| `x → c`             | The independent variable `x` is approaching a specific value `c`.  | Our time interval `Δt` is approaching `0`. `Δt → 0`                         |
| `f(x)`              | The function whose value we are observing.                         | Our average speed formula: `(p(2+Δt) - p(2)) / Δt`                            |
| `L`                 | The Limit. The value that `f(x)` gets infinitely close to.         | The value our average speed approached: `4`.                                |

So, the precise mathematical statement for our problem is:
`lim (Δt→0) [ (2+Δt)² - 2² ] / Δt = 4`

In the next section, we will generalize this expression into the formal definition of the derivative and learn how to solve it algebraically.

## **Section 2: The Derivative - Formalizing Instantaneous Change**

#### The Big Picture
In the last section, we built the intuition that the instantaneous speed at `t=2` is the *limit* of the average speed as the time interval shrinks to zero. Now, we will generalize this process into a single, powerful formula. This formula, called the **definition of the derivative**, will allow us to take any function `f(x)` and derive a new function, `f'(x)`, that describes its instantaneous rate of change at *every single point*.

#### From a Specific Point to a General Function
Our calculation in Section 1 was specific to the point `t=2`.
`lim (Δt→0) [ p(2+Δt) - p(2) ] / Δt`

To create a function for the rate of change that works for *any* time `t`, we simply replace the specific point `2` with the general variable `t`.

Let's also adopt standard mathematical notation:
*   We will use `x` instead of `t` for the general variable.
*   We will use `h` instead of `Δt` for the small change.
*   The function is `f(x)` instead of `p(t)`.

This gives us the formal definition of the derivative.

#### The Formula: The Derivative from First Principles

The derivative of a function `f(x)`, denoted as `f'(x)` (read "f prime of x"), is defined as:

`f'(x) = lim (h→0) [f(x+h) - f(x)] / h`

This is the most important formula in differential calculus. Every single rule you will learn is derived from it.

| Component         | Meaning                                                                 | Our Car Example `p(t) = t²`                                |
| ----------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------- |
| `h`               | A very small change in the input (e.g., a tiny slice of time).          | `Δt`, a small change in time.                              |
| `x`               | The point at which we are evaluating the rate of change.                | `t`, any specific time.                                    |
| `f(x)`            | The original function's value at `x`.                                   | `p(t) = t²`, the position at time `t`.                     |
| `f(x+h)`          | The function's value a tiny bit *after* `x`.                            | `p(t+h) = (t+h)²`, the position a moment later.           |
| `f(x+h) - f(x)`   | The change in the function's value (the "rise").                        | `p(t+h) - p(t)`, the change in position.                   |
| `[...]/h`         | The average rate of change over that tiny interval (the "rise over run"). | The average speed over the tiny time interval `h`.         |
| `lim (h→0)`       | The limit as that interval shrinks to zero.                             | Finding the instantaneous speed from the average speed.    |
| `f'(x)`           | **The result:** a new function for the instantaneous rate of change.    | `v(t)`, a new function for the velocity (speed) at any time `t`. |

#### Step-by-Step Example: Deriving the Derivative of `f(x) = x²`

Let's use the formula to prove the "magic" from the introduction. We will find `f'(x)` for `f(x) = x²`.

1.  **Set up the formula.**
    Substitute `f(x) = x²` into the definition:
    `f'(x) = lim (h→0) [ (x+h)² - x² ] / h`

2.  **Expand the algebraic term in the numerator.**
    ` (x+h)² = x² + 2xh + h² `
    Our expression becomes:
    `f'(x) = lim (h→0) [ (x² + 2xh + h²) - x² ] / h`

3.  **Simplify the numerator.**
    The `x²` terms cancel each other out. This will almost always happen.
    `f'(x) = lim (h→0) [ 2xh + h² ] / h`

4.  **Factor `h` out of the numerator and cancel.**
    Before this step, plugging in `h=0` would give an undefined `0/0` result. By factoring, we remove this problem.
    `f'(x) = lim (h→0) [ h(2x + h) ] / h`
    Now cancel the `h` from the top and bottom:
    `f'(x) = lim (h→0) [ 2x + h ]`

5.  **Evaluate the limit.**
    Now that `h` is no longer in the denominator, we can safely let `h` become 0.
    `f'(x) = 2x + 0`
    `f'(x) = 2x`

#### The Result and Connection
We have rigorously proven that if `f(x) = x²`, then its derivative is `f'(x) = 2x`.

Let's apply this back to our running example:
*   The car's position function is `p(t) = t²`.
*   Its velocity (speed) function is the derivative, `v(t) = p'(t) = 2t`.
*   To find the instantaneous speed at `t=2`, we now use our new velocity function:
    `v(2) = 2 * 2 = 4` m/s.

This matches the result we discovered intuitively with the table of narrowing intervals. We now have a formal method for finding the rate of change for any point `t`, not just `t=2`.

Using this "first principles" definition is powerful, but it is slow. In the next section, we will learn shortcuts to make this process immediate.

## **Section 3: The Library - Derivatives of Essential Functions**

#### The Big Picture
We have our universal tool: the limit definition of the derivative. Now, we will use it to build our library of essential derivatives.

Our strategy will be practical:
1.  For simple functions, we will step through the full proof to practice the limit definition.
2.  For more complex functions (like `sin(x)` or `ln(x)`), the formal proofs are long and rely on other advanced concepts. We will skip them. Instead, we will state the result and build a deep graphical intuition so you understand *why* the result is correct.

#### Simple Functions: Constants and Lines

**1. The Derivative of a Constant: `f(x) = c`**
*   **Intuition:** The graph of `f(x) = 5` is a flat, horizontal line. Its slope is always `0`. The derivative *is* the slope, so the derivative must be `0`.
*   **Proof:** We can confirm this easily.
    *   `f'(x) = lim (h→0) [ f(x+h) - f(x) ] / h`
    *   `f'(x) = lim (h→0) [ c - c ] / h`
    *   `f'(x) = lim (h→0) [ 0 ] / h = 0`
*   **Result:** `d/dx(c) = 0`

**2. The Derivative of a Line: `f(x) = mx` (e.g., `f(x)=2x`)**
*   **Intuition:** The graph of `f(x) = mx` is a straight line passing through the origin with a constant slope of `m`. The derivative should be `m`.
*   **Proof:**
    *   `f'(x) = lim (h→0) [ m(x+h) - mx ] / h`
    *   `f'(x) = lim (h→0) [ mx + mh - mx ] / h`
    *   `f'(x) = lim (h→0) [ mh ] / h = m`
*   **Result:** `d/dx(mx) = m`

#### The Natural Exponential: `f(x) = eˣ`

The number `e` (≈ 2.718) is the most important base for an exponential function in calculus. It has a unique and remarkable property.

*   **Result:** The function `f(x) = eˣ` is its own derivative.
    `d/dx(eˣ) = eˣ`
*   **Graphical Intuition:** This means that for the graph of `y = eˣ`, the **slope of the tangent line at any point `x` is equal to the value of the function `y` at that same point.**
    *   **Function to Plot:** A graph of `y = eˣ`. It's an exponential growth curve that passes through the point `(0, 1)`.

| At Point `x` | Function Value `y = eˣ` | Slope of Tangent Line | Observation           |
| :----------: | :----------------------: | :--------------------: | --------------------- |
|    `x=0`     |        `e⁰ = 1`        |         `m = 1`        | The slope equals the value. |
|    `x=1`     |      `e¹ ≈ 2.718`      |      `m ≈ 2.718`       | The slope equals the value. |
|    `x=-1`    |     `e⁻¹ ≈ 0.367`      |      `m ≈ 0.367`       | The slope equals the value. |

This is the only function (aside from `f(x)=0`) with this property, which is why `e` is so fundamental to calculus.

#### The Natural Logarithm: `f(x) = ln(x)`

The natural logarithm is the inverse of `eˣ`. Its derivative is also clean and fundamental.

*   **Result:**
    `d/dx(ln(x)) = 1/x`
*   **Graphical Intuition:** The function `1/x` perfectly describes the slope of the `ln(x)` graph.
    *   **Function to Plot:** A graph of `y = ln(x)`. It rises from negative infinity, crosses the x-axis at `(1, 0)`, and grows slowly.

Let's check the slope of `ln(x)` at different points and see if it matches the value of `1/x`.

| At Point `x` | Slope of `ln(x)` Graph                  | Value of Derivative `1/x` | Observation     |
| :----------: | :--------------------------------------: | :------------------------: | --------------- |
|  `x=0.1`   | Very steep, positive slope              |        `1/0.1 = 10`        | Matches (large) |
|    `x=1`     | Moderately steep, slope looks like `1`  |         `1/1 = 1`          | Matches         |
|    `x=10`    | Very shallow, slope is close to zero    |        `1/10 = 0.1`        | Matches (small) |

#### Trigonometric Functions: `sin(x)` and `cos(x)`

The derivatives of sine and cosine are elegantly linked. We will skip the formal proofs and demonstrate their relationship graphically.

*   **Result 1:** `d/dx(sin(x)) = cos(x)`
*   **Result 2:** `d/dx(cos(x)) = -sin(x)`

*   **Graphical Intuition for `d/dx(sin(x)) = cos(x)`**
    The `cos(x)` function acts as a "slope map" for the `sin(x)` function.
    *   **Functions to Plot:** `y = sin(x)` on top, and `y = cos(x)` directly below it, aligned on the x-axis.

| At Point `x` | On the `sin(x)` Graph...                | The Slope Is... | On the `cos(x)` Graph, the Value Is... |
| :----------: | :-------------------------------------- | :--------------: | :-----------------------------------: |
|    `x=0`     | The graph is rising most steeply.       |        `+1`        |                  `1`                  |
|   `x=π/2`    | The graph is at its peak (max value).   |       `0`        |                  `0`                  |
|    `x=π`     | The graph is falling most steeply.      |       `-1`       |                 `-1`                  |
|  `x=3π/2`  | The graph is at its trough (min value). |       `0`        |                  `0`                  |

The value of `cos(x)` at every point perfectly predicts the slope of `sin(x)` at that same point. The other derivatives, like for `tan(x)`, can be found using the rules we will learn in the next sections.


## **Section 4: The Toolkit - Basic Differentiation Rules**

#### The Big Picture
Using the limit definition from first principles is the correct way to *understand* the derivative, but it is slow and inefficient for actually *calculating* derivatives. We will now learn three fundamental rules that are derived from the limit definition. These rules will allow us to differentiate any polynomial function in seconds, without ever writing the word "lim" again.

#### The Motivating Problem
Consider a more realistic position function for an object:
`p(t) = 4t³ - 5t² + 7t + 10`

Calculating the derivative of this function using the limit definition would involve expanding `4(t+h)³`, `-5(t+h)²`, and so on. The algebra would be extremely tedious and error-prone. We need a better way.

Let's build the toolkit to solve this problem instantly.

---

### **Rule 1: The Power Rule**
This is the most common and powerful rule you will use. It applies to any term of the form `xⁿ`.

*   **For functions like:** `x²`, `x³`, `x¹⁰⁰`, `√x`, `1/x³`
*   **Formula:** `d/dx (xⁿ) = nxⁿ⁻¹`
*   **Algorithm:**
    1.  Bring the old exponent down as a multiplier.
    2.  Subtract one from the old exponent to get the new exponent.

**Examples:**

| Input `f(x)` | Rewritten as `xⁿ` | Apply Power Rule `nxⁿ⁻¹` | Result `f'(x)` |
| :----------- | :--------------: | :-----------------------: | :-------------: |
| `x²`         |       `x²`       |         `2x²⁻¹`         |      `2x`       |
| `x⁵`         |       `x⁵`       |         `5x⁵⁻¹`         |      `5x⁴`      |
| `x`          |       `x¹`       |         `1x¹⁻¹`         | `1x⁰ = 1`     |
| `√x`         |     `x¹/²`     |      `(1/2)x¹/²⁻¹`      | `(1/2)x⁻¹/²`  |
| `1/x²`       |       `x⁻²`      |        `-2x⁻²⁻¹`        |     `-2x⁻³`     |

---

### **Rule 2: The Constant Multiple Rule**
This rule explains how to handle constants that are multiplied by a function.

*   **For functions like:** `5x³`, `10x²`, `-7sin(x)`
*   **Formula:** `d/dx [c * f(x)] = c * f'(x)`
*   **Algorithm:** The constant is unaffected and "comes along for the ride." You simply differentiate the function part and keep the constant as a multiplier.

**Examples:**
*   **Problem:** Find the derivative of `f(x) = 5x³`.
    *   **Input:** `f(x) = 5 * x³`
    *   **Identify:** `c=5`, `f(x)=x³`.
    *   **Differentiate `f(x)`:** `d/dx(x³) = 3x²`.
    *   **Multiply by `c`:** `5 * (3x²) = 15x²`.
    *   **Output:** `d/dx(5x³) = 15x²`

*   **Problem:** Find the derivative of `g(x) = -2cos(x)`.
    *   **Input:** `g(x) = -2 * cos(x)`
    *   **Differentiate `cos(x)`:** `d/dx(cos(x)) = -sin(x)`.
    *   **Multiply by constant:** `-2 * (-sin(x)) = 2sin(x)`.
    *   **Output:** `d/dx(-2cos(x)) = 2sin(x)`

---

### **Rule 3: The Sum/Difference Rule**
This rule allows us to break complex functions into simpler parts.

*   **For functions like:** `x² + 5x³`, `sin(x) - eˣ`
*   **Formula:** `d/dx [f(x) ± g(x)] = f'(x) ± g'(x)`
*   **Algorithm:** If your function is a chain of terms added or subtracted, you can differentiate each term individually and then combine the results.

**Example:**
*   **Problem:** Find the derivative of `h(x) = x² - 7x`.
    *   **Input:** `h(x) = x² - 7x`
    *   **Differentiate term 1 (`x²`):** `d/dx(x²) = 2x`.
    *   **Differentiate term 2 (`-7x`):** `d/dx(-7x) = -7 * d/dx(x) = -7 * 1 = -7`.
    *   **Combine results:** `2x - 7`.
    *   **Output:** `d/dx(x² - 7x) = 2x - 7`.

---

### **Solving the Motivating Problem**
Now, let's use all three rules to instantly find the velocity function `v(t) = p'(t)` for `p(t) = 4t³ - 5t² + 7t + 10`.

`p'(t) = d/dt (4t³ - 5t² + 7t + 10)`

1.  **Apply the Sum/Difference Rule:** We can differentiate term-by-term.
    `p'(t) = d/dt(4t³) - d/dt(5t²) + d/dt(7t) + d/dt(10)`

2.  **Differentiate each term:**
    *   `d/dt(4t³)` = `4 * (3t²) = 12t²` (Constant Multiple + Power Rule)
    *   `d/dt(5t²)` = `5 * (2t¹) = 10t` (Constant Multiple + Power Rule)
    *   `d/dt(7t)` = `7 * (1t⁰) = 7` (Constant Multiple + Power Rule)
    *   `d/dt(10)` = `0` (Constant Rule from Section 3)

3.  **Combine the results:**
    `v(t) = p'(t) = 12t² - 10t + 7`

What would have taken dozens of lines of algebra with the limit definition, we have now solved in one line. This is the power of the basic differentiation rules.

## **Section 5: Advanced Toolkit - Product and Quotient Rules**

#### The Big Picture
We can now differentiate functions added together, like `x² + sin(x)`. But what happens when functions are multiplied, like `x² * sin(x)`, or divided, like `x² / sin(x)`? We need specific rules for these operations.

A common mistake is to think the derivative of `f(x)g(x)` is simply `f'(x)g'(x)`. This is incorrect. For example, if `h(x) = x * x = x²`, we know the correct derivative is `h'(x) = 2x`. But the incorrect method would give `d/dx(x) * d/dx(x) = 1 * 1 = 1`, which is wrong.

This section provides the correct formulas for handling products and quotients.

---

### **Rule 4: The Product Rule**
This rule is used to find the derivative of two functions multiplied together.

*   **Formula:** `d/dx [f(x)g(x)] = f'(x)g(x) + f(x)g'(x)`
*   **Mnemonic:** "The derivative of the first, times the second, plus the first, times the derivative of the second."

#### The Algorithm
To find the derivative of a product, follow these steps:
1.  Identify the two functions being multiplied, `f(x)` and `g(x)`.
2.  Find their individual derivatives, `f'(x)` and `g'(x)`.
3.  Assemble the four pieces according to the Product Rule formula.
4.  Simplify the resulting expression.

#### Step-by-Step Example
**Problem:** Find the derivative of `h(x) = x² sin(x)`.

1.  **Identify `f(x)` and `g(x)`.**
    *   `f(x) = x²`
    *   `g(x) = sin(x)`

2.  **Find their derivatives.**
    *   `f'(x) = 2x` (using the Power Rule)
    *   `g'(x) = cos(x)` (from our function library)

3.  **Assemble the pieces using the formula `f'(x)g(x) + f(x)g'(x)`.**
    `h'(x) = (2x) * (sin(x)) + (x²) * (cos(x))`

4.  **Simplify.**
    `h'(x) = 2x sin(x) + x² cos(x)`

This is the final derivative. There is no simpler way to write it.

---

### **Rule 5: The Quotient Rule**
This rule is used to find the derivative of one function divided by another.

*   **Formula:** `d/dx [f(x) / g(x)] = [f'(x)g(x) - f(x)g'(x)] / [g(x)]²`
*   **Mnemonic:** "Low D-High minus High D-Low, over the square of what's below."
    *   "Low" refers to the bottom function, `g(x)`.
    *   "High" refers to the top function, `f(x)`.
    *   "D-High" means the derivative of the top, `f'(x)`.
    *   "D-Low" means the derivative of the bottom, `g'(x)`.

#### The Algorithm
1.  Identify the "High" function `f(x)` and the "Low" function `g(x)`.
2.  Find their individual derivatives, `f'(x)` and `g'(x)`.
3.  Assemble the pieces according to the Quotient Rule formula.
4.  Simplify.

#### Step-by-Step Example
**Problem:** Find the derivative of `h(x) = eˣ / x³`.

1.  **Identify High and Low functions.**
    *   High: `f(x) = eˣ`
    *   Low: `g(x) = x³`

2.  **Find their derivatives.**
    *   `f'(x) = eˣ`
    *   `g'(x) = 3x²`

3.  **Assemble the pieces using `[f'g - fg'] / g²`.**
    `h'(x) = [ (eˣ) * (x³) - (eˣ) * (3x²) ] / (x³)²`

4.  **Simplify.**
    *   Simplify the denominator: `(x³)² = x⁶`.
    *   Factor the numerator. The common term is `eˣx²`.
        `h'(x) = [ eˣx²(x - 3) ] / x⁶`
    *   Cancel `x²` from the top and bottom.
        `h'(x) = [ eˣ(x - 3) ] / x⁴`

This is the final derivative.

#### A Practical Application: Deriving `tan(x)`
We can now prove the derivative of `tan(x)` from our library. We know that `tan(x) = sin(x) / cos(x)`.

*   High: `f(x) = sin(x)`  ->  `f'(x) = cos(x)`
*   Low: `g(x) = cos(x)`  ->  `g'(x) = -sin(x)`

Using the Quotient Rule:
`d/dx(tan(x)) = [ (cos(x))(cos(x)) - (sin(x))(-sin(x)) ] / (cos(x))²`
`= [ cos²(x) + sin²(x) ] / cos²(x)`

Using the Pythagorean identity `sin²(x) + cos²(x) = 1`:
`= 1 / cos²(x)`

Since `1/cos(x) = sec(x)`, the result is `sec²(x)`. We have successfully derived a rule from our library using a combination of other rules.

We can now handle functions combined with `+`, `-`, `*`, and `/`. The final and most important rule of combination is how to handle functions *inside* of other functions. That is the subject of the next section: the Chain Rule.

## **Section 6: The Master Key - The Chain Rule**

#### The Big Picture
The Chain Rule is the rule for differentiating **composite functions**—a function nested inside of another function. It is the final and most powerful tool for combining derivatives.

#### The Formal Definition
If a function `h(x)` is a composition of two functions, written as `h(x) = f(g(x))`, its derivative is:

`h'(x) = f'(g(x)) * g'(x)`

**The Algorithm:**
1.  Differentiate the **outer function `f`** while keeping the **inner function `g(x)`** unchanged inside of it.
2.  Multiply the result by the derivative of the **inner function `g(x)`**.

In Leibniz notation, if `y = f(u)` and `u = g(x)`, the rule is:
`dy/dx = dy/du * du/dx`

Let's apply this immediately.

---
#### Motivating Example 1: A Polynomial Power
**Problem:** Find the derivative of `h(x) = (x³ + 5x)⁷`.
Attempting to expand this algebraically would be a nightmare. The Chain Rule solves it instantly.

1.  **Identify Outer and Inner Functions.**
    *   **Outer function `f(...)`:** `(something)⁷`. Its derivative is `7(something)⁶`.
    *   **Inner function `g(x)`:** `x³ + 5x`. Its derivative is `3x² + 5`.

2.  **Apply the Chain Rule `f'(g(x)) * g'(x)`.**
    *   **Step 1: Differentiate the outer function, keeping the inner part.**
        `f'(g(x)) = 7(x³ + 5x)⁶`
    *   **Step 2: Find the derivative of the inner function.**
        `g'(x) = 3x² + 5`
    *   **Step 3: Multiply them together.**
        `h'(x) = 7(x³ + 5x)⁶ * (3x² + 5)`

This is the complete derivative. The process is clean and avoids any complex algebra.

---
#### Motivating Example 2: A Trigonometric Function
**Problem:** Find the derivative of `h(x) = sin(x²)`.
This is *not* `sin(x) * sin(x)`. It is a single function where the input to `sin` is `x²`.

1.  **Identify Outer and Inner Functions.**
    *   **Outer function `f(...)`:** `sin(something)`. Its derivative is `cos(something)`.
    *   **Inner function `g(x)`:** `x²`. Its derivative is `2x`.

2.  **Apply the Chain Rule.**
    *   **Derivative of outer (keeping inner):** `cos(x²)`.
    *   **Derivative of inner:** `2x`.
    *   **Multiply:** `h'(x) = cos(x²) * 2x`.

    It is conventional to write the polynomial part first:
    `h'(x) = 2x cos(x²)`.

---
#### Motivating Example 3: An Exponential Function
**Problem:** Find the derivative of a bacterial population modeled by `P(t) = 50e⁰.²ᵗ`.

1.  **Identify Outer and Inner Functions.**
    *   The constant `50` comes along for the ride (Constant Multiple Rule).
    *   **Outer function `f(...)`:** `e^(something)`. Its derivative is `e^(something)`.
    *   **Inner function `g(t)`:** `0.2t`. Its derivative is `0.2`.

2.  **Apply the Chain Rule.**
    *   **Derivative of outer (keeping inner):** `e⁰.²ᵗ`.
    *   **Derivative of inner:** `0.2`.
    *   **Multiply everything, including the constant:**
        `P'(t) = 50 * (e⁰.²ᵗ) * (0.2)`

3.  **Simplify.**
    `P'(t) = 10e⁰.²ᵗ`

**Interpretation:** The derivative `P'(t)` represents the instantaneous *growth rate* of the bacteria population. We can now calculate this rate at any time `t`. For example, at `t=10` hours, the growth rate is `P'(10) = 10e²` (approximately 74 bacteria per hour).

---
#### Connecting to Reality: The Expanding Ripple
Imagine dropping a stone in a pond. A circular ripple expands outwards.
*   The radius of the ripple, `r`, is growing at a constant rate of **2 cm per second**. This is `dr/dt = 2`.
*   The Area of the circle is given by the formula `A = πr²`. This links Area to radius.

**Question:** How fast is the **Area** of the circle growing with respect to **time** (`dA/dt`) when the radius is `r = 10` cm?

Here, `A` is a function of `r`, and `r` is a function of `t`. This is a perfect scenario for the Chain Rule.

1.  **Identify the rates we have.**
    *   `dA/dr = 2πr` (Derivative of Area with respect to radius).
    *   `dr/dt = 2` (Derivative of radius with respect to time).

2.  **Apply the Chain Rule `dA/dt = dA/dr * dr/dt`.**
    `dA/dt = (2πr) * (2) = 4πr`

3.  **Interpret the Result.**
    This new function, `dA/dt = 4πr`, gives the growth rate of the area at any given radius `r`.

4.  **Answer the specific question.**
    At the moment when `r = 10` cm:
    `dA/dt = 4π(10) = 40π` cm²/s.
    The area is increasing at a rate of approximately `125.6` cm² per second.

The Chain Rule allows us to link related rates of change in a precise mathematical way.

## **Section 7: Implicit Differentiation - Deriving When You Can't Solve for y**

#### The Big Picture
So far, every function we've differentiated has been **explicitly** defined, meaning it's in the form `y = f(x)`. We could always get `y` by itself on one side of the equation.

But many important relationships are defined **implicitly**, where `x` and `y` are mixed together, such as the equation of a circle: `x² + y² = 25`.

Trying to solve for `y` is messy (`y = ±√(25 - x²)`), giving two separate functions to differentiate. Implicit differentiation is a technique that lets us find the derivative `dy/dx` directly from the original equation, without ever solving for `y`. It is not a new rule, but a clever application of the Chain Rule.

#### The Core Concept
The key is to assume that `y` is a function of `x` (we can write it as `y(x)`). Then, whenever we differentiate a term containing `y`, we must apply the Chain Rule.

Let's compare:
*   `d/dx (x²) = 2x` (Standard Power Rule)
*   `d/dx (y²) = ?`
    *   **Outer function:** `(something)²`. Its derivative is `2 * (something)`.
    *   **Inner function:** `y`. Its derivative is `dy/dx`.
    *   **Result (by Chain Rule):** `d/dx (y²) = 2y * dy/dx`

This is the central idea. Every time you differentiate a `y` term, you must multiply by `dy/dx`.

#### The Algorithm for Implicit Differentiation
1.  Take the derivative of **both sides** of the equation with respect to `x`.
2.  When differentiating terms with `y`, apply the Chain Rule by multiplying by `dy/dx`.
3.  After differentiating, you will have an equation that contains `x`, `y`, and `dy/dx`.
4.  Solve this equation algebraically for `dy/dx`.

---
#### Motivating Example 1: The Slope of a Circle
**Problem:** Find the formula for the slope (`dy/dx`) of the circle `x² + y² = 25`. Then, find the slope at the point `(3, 4)`.

1.  **Differentiate both sides with respect to `x`.**
    `d/dx (x² + y²) = d/dx (25)`

2.  **Apply the rules term-by-term.**
    *   `d/dx (x²) = 2x`
    *   `d/dx (y²) = 2y * dy/dx` (This is the crucial Chain Rule step)
    *   `d/dx (25) = 0` (Derivative of a constant)
    The equation becomes:
    `2x + 2y * dy/dx = 0`

3.  **Solve for `dy/dx`.**
    *   `2y * dy/dx = -2x`
    *   `dy/dx = -2x / 2y`
    *   `dy/dx = -x / y`

**Result and Interpretation:**
We found that the slope of the tangent line at any point `(x, y)` on the circle is given by the simple formula `dy/dx = -x/y`. Unlike explicit derivatives, the slope here depends on both the `x` and `y` coordinates. This makes perfect sense for a circle.

*   **Find the slope at `(3, 4)`:**
    `dy/dx` at `(3, 4)` is `-3 / 4`. The slope is -0.75.
*   **What about the point `(0, 5)`?**
    `dy/dx = -0 / 5 = 0`. The tangent line is horizontal, which is correct for the top of the circle.
*   **What about the point `(5, 0)`?**
    `dy/dx = -5 / 0`. The slope is undefined. The tangent line is vertical, which is correct for the rightmost point of the circle.

---
#### Motivating Example 2: Deriving the Derivative of `arcsin(x)`
We can now use this technique to derive the formulas for inverse functions.

**Problem:** Find the derivative of `y = arcsin(x)`.

1.  **Rewrite the function implicitly.**
    If `y = arcsin(x)`, then `sin(y) = x`. This form is easier to work with.

2.  **Differentiate both sides with respect to `x`.**
    `d/dx (sin(y)) = d/dx (x)`

3.  **Apply the rules.**
    *   `d/dx (sin(y)) = cos(y) * dy/dx` (Chain Rule on the `y` term)
    *   `d/dx (x) = 1`
    The equation becomes:
    `cos(y) * dy/dx = 1`

4.  **Solve for `dy/dx`.**
    `dy/dx = 1 / cos(y)`

5.  **Express the result in terms of `x`.**
    Our answer is correct, but it's in terms of `y`. We need it in terms of `x`. We use a right-triangle trick based on our original relationship, `sin(y) = x = x/1`.
    *   Draw a right triangle and label angle `y`.
    *   Since `sin(y) = Opposite / Hypotenuse`, label the Opposite side `x` and the Hypotenuse `1`.
    *   By the Pythagorean theorem (`a² + b² = c²`), the Adjacent side is `√(1² - x²) = √(1 - x²)`.
    *   Now, find `cos(y)` from the triangle: `cos(y) = Adjacent / Hypotenuse = √(1 - x²) / 1`.

6.  **Substitute back into our derivative.**
    `dy/dx = 1 / cos(y) = 1 / √(1 - x²)`

**Result:** We have proven that `d/dx(arcsin(x)) = 1 / √(1 - x²)`, adding another powerful tool to our derivative library.

## **Section 8: Conclusion and Next Steps**

This tutorial has systematically deconstructed differential calculus, from its foundational limit definition to the rules required for practical application. We began with a single problem—finding the instantaneous velocity from a position function `p(t) = t²`—and built a complete framework to solve not only that problem but any derivative of the functions listed below.

You now possess the complete toolkit for single-variable differentiation.

#### The Completed Toolbox

| Category                | Function `f(x)`       | Derivative `f'(x)`                |
| ----------------------- | --------------------- | --------------------------------- |
| **Basic Rules**         | `c` (constant)        | `0`                               |
|                         | `xⁿ`                  | `nxⁿ⁻¹`                           |
| **Essential Functions** | `eˣ`                  | `eˣ`                              |
|                         | `ln(x)`               | `1/x`                             |
| **Trigonometric**       | `sin(x)`              | `cos(x)`                          |
|                         | `cos(x)`              | `-sin(x)`                         |
|                         | `tan(x)`              | `sec²(x)`                         |
| **Inverse Trig**        | `arcsin(x)`           | `1 / √(1 - x²)`                   |
|                         | `arctan(x)`           | `1 / (1 + x²)`                    |
| **Combination Rules**   | `f(x) ± g(x)`         | `f'(x) ± g'(x)` (Sum Rule)        |
|                         | `f(x)g(x)`            | `f'(x)g(x) + f(x)g'(x)` (Product Rule) |
|                         | `f(x) / g(x)`         | `[f'g - fg'] / g²` (Quotient Rule)  |
|                         | `f(g(x))`             | `f'(g(x)) * g'(x)` (Chain Rule)     |

#### What to Learn Next

Mastering the derivative is the first of two major pillars of calculus. Your knowledge is now the foundation for these subsequent topics:

1.  **Applications of the Derivative:**
    *   **Optimization:** Using the fact that `f'(x) = 0` at local maxima and minima to solve problems that require maximizing or minimizing a quantity (e.g., maximum profit, minimum material cost).
    *   **Related Rates:** Solving more complex versions of the "expanding ripple" problem, where the rates of change of multiple variables are linked.
    *   **L'Hôpital's Rule:** A technique that uses derivatives to evaluate limits that result in indeterminate forms like `0/0` or `∞/∞`.

2.  **Integral Calculus:**
    *   This is the second pillar of calculus and represents the inverse operation of differentiation.
    *   **The Problem:** While the derivative finds the instantaneous *rate of change* (slope) of a function, the integral finds the cumulative *total* (area under the curve).
    *   **The Fundamental Theorem of Calculus:** This critical theorem formally links differentiation and integration, showing they are inverse operations.

3.  **Advanced Topics:**
    *   **Series and Sequences:** Using calculus to analyze the behavior of infinite sums.
    *   **Multivariable Calculus:** Extending the concepts of derivatives and integrals to functions of multiple variables (e.g., `f(x, y, z)`), which is essential for 3D physics, machine learning, and economics.