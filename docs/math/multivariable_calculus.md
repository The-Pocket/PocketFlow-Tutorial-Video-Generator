# **Title: Give me 1 hour, I will make multivariable calculus click forever**

## **Introduction: The Promise - From Flatland to Spaceland**

In single-variable calculus, you mastered the world of `y = f(x)`—curves on a flat plane. You learned to find the slope at a point (`dy/dx`) and the area under the curve (`∫f(x)dx`).

Now, we move to higher dimensions. This is not just adding a variable; it is a fundamental shift in perspective. We are moving from the 2D world of curves to the 3D world of surfaces, like `z = f(x, y)`, and beyond. The questions we ask become richer and more physical:

*   Instead of "how steep is the slope?", we ask, **"Which direction is the steepest?"**
*   Instead of finding a maximum on a curve, we ask, **"Where is the peak of this mountain?"**
*   Instead of calculating area, we ask, **"What is the volume under this surface?"**

**The Promise:** In the next hour, you will learn that the tools of calculus generalize beautifully to answer these questions. Derivatives become vectors (`∇f`, the Gradient). Second derivatives become matrices (`H`, the Hessian). You will master the core tools for optimization, coordinate transformations, and understanding vector fields—the language of physics, machine learning, and computer graphics.

This table is the core of multivariable calculus. By the end of this tutorial, you will understand what each tool is and how to use it.

**The Toolbox You Will Master**

| Technique/Concept              | Notation / Formula                                     | What It Solves                                                                |
| ------------------------------ | ------------------------------------------------------ | ----------------------------------------------------------------------------- |
| **The Gradient**               | `∇f = <∂f/∂x, ∂f/∂y>`                                  | Finds the direction of steepest ascent on a surface.                            |
| **The Jacobian**               | `J` (matrix of partials)                               | Translates how changes in inputs affect outputs in multiple dimensions.         |
| **The Hessian**                | `H` (matrix of 2nd partials)                           | Determines the shape of a surface: peak, valley, or saddle.                     |
| **Unconstrained Optimization** | `∇f = 0`, then test `H`                                | Finds and classifies the maximums and minimums of a function.                 |
| **Lagrange Multipliers**       | `∇f = λ∇g`                                             | Finds the maximum/minimum of a function subject to a constraint.              |
| **Double Integrals**           | `∬ f(x,y) dA`                                          | Calculates the volume under a surface or the mass of a variable-density plate. |
| **Change of Variables**        | `dx dy = |det(J)| du dv`                                | Simplifies integrals by switching to a more natural coordinate system (like polar). |
| **Divergence & Curl**          | `∇ ⋅ F` (div), `∇ × F` (curl)                          | Measures the "outward flow" and "rotational spin" of a vector field.            |

Let's begin.

## **Part 1: The Geometry of Change - Derivatives in Higher Dimensions**

---

### **Section 1: The Gradient (∇f) - Your Compass on a Mountain**

#### The Big Picture

We are standing on a surface in 3D space. Finding "the slope" is ambiguous because it depends on which direction you walk. To solve this, we start with the simplest possible directions: due East (the positive x-direction) and due North (the positive y-direction). The slopes in these two special directions are called **partial derivatives**. Once we have them, we can assemble them into a vector called the **gradient**, which will then tell us the slope in *any* direction.

#### The Motivating Problem

You are standing on a hill described by the height function `h(x, y) = 100 - x² - 2y²`. You are at the point `(x=1, y=1)`. The height here is `h(1, 1) = 97`.

Our goal: Find the direction of steepest ascent and the slope in that direction.

#### The Core Idea: Slicing the Mountain

Before we can find the steepest slope, let's find the slope in just one direction: the x-direction (East).

Imagine you are physically at `(1, 1)` and you decide to walk *only* along the East-West line. This means your `y`-coordinate is frozen. It is fixed at `y=1`.

If `y=1` is constant, let's plug it into our height function:
`h(x, y) = 100 - x² - 2y²`
`h(x, for y=1) = 100 - x² - 2(1)²`
`h(x, for y=1) = 100 - x² - 2 = 98 - x²`

Look what happened! By freezing `y=1`, our 3D surface problem has collapsed into a 2D curve problem: `f(x) = 98 - x²`. This is just a simple parabola. You already know how to find its slope from single-variable calculus: take the derivative.

*   **Function of the slice:** `f_slice(x) = 98 - x²`
*   **Derivative of the slice:** `f'_slice(x) = -2x`
*   **Slope at our point `x=1`:** `f'_slice(1) = -2(1) = -2`.

The slope of our hill, specifically in the x-direction at `(1, 1)`, is **-2**.

This process—freezing one variable to turn a surface into a curve and then taking a regular derivative—is called a **partial derivative**.

#### Formalizing Partial Derivatives

The shortcut is to "treat the other variable as a constant" *when you take the derivative*.

Let's re-calculate the slope in the x-direction using this rule.
**Function:** `h(x, y) = 100 - x² - 2y²`
**Goal:** Find the partial derivative with respect to x, written `∂h/∂x`.

| Term        | Derivative with respect to `x` | Why?                                                                |
| ----------- | ----------------------------- | ------------------------------------------------------------------- |
| `100`       | `0`                           | The derivative of any constant is 0.                                |
| `-x²`       | `-2x`                         | This is the variable we are differentiating.                         |
| `-2y²`      | `0`                           | We are treating `y` as a constant. So `2y²` is also just a constant. |

Combining these gives:
`∂h/∂x = 0 - 2x - 0 = -2x`.

This is the general formula for the slope in the x-direction at any point `(x, y)`. At our specific point `(1, 1)`, the slope is `∂h/∂x |_(1,1) = -2(1) = -2`, which matches our slicing method.

Now let's do the same for the y-direction (`∂h/∂y`), treating `x` as a constant.
**Function:** `h(x, y) = 100 - x² - 2y²`

| Term        | Derivative with respect to `y` | Why?                                                                |
| ----------- | ----------------------------- | ------------------------------------------------------------------- |
| `100`       | `0`                           | The derivative of a constant is 0.                                  |
| `-x²`       | `0`                           | We are treating `x` as a constant. So `x²` is also just a constant. |
| `-2y²`      | `-4y`                         | This is the variable we are differentiating.                         |

Combining these gives:
`∂h/∂y = 0 - 0 - 4y = -4y`.
At our point `(1, 1)`, the slope in the y-direction is `∂h/∂y |_(1,1) = -4(1) = -4`.

#### The Gradient: Assembling the Slopes

We now have the two fundamental slopes at `(1, 1)`:
*   Slope in x-direction: -2
*   Slope in y-direction: -4

The **gradient** is simply the vector that packages these two pieces of information together. It is written `∇h` (pronounced "del h").

**The Formula:**
`∇f(x, y) = < ∂f/∂x, ∂f/∂y >`

For our hill, the general gradient is `∇h = <-2x, -4y>`. At our specific point `(1, 1)`, the gradient is:
`∇h(1, 1) = < -2, -4 >`

This vector answers our original questions.
1.  **Direction of Steepest Ascent:** The gradient vector `<-2, -4>` points directly in the direction of the steepest path up the hill.
2.  **Steepness of that Path:** The magnitude (length) of the gradient vector gives the actual slope in that steepest direction.
    `Slope = ||∇h(1, 1)|| = ||< -2, -4 >|| = √((-2)² + (-4)²) = √(4 + 16) = √20 ≈ 4.47`

The steepest slope at this point is approximately **4.47**. The direction to get that slope is `< -2, -4 >`. The gradient is your compass and your slope-meter, all in one.

## **Section 2: The Jacobian (J) - The Universal Translator for Derivatives**

#### The Big Picture

In the last section, we studied functions with multiple inputs but only one output, like `height = h(x, y)`. The derivative was a vector, the gradient `∇h`.

Now, we consider functions with **multiple inputs and multiple outputs**. For example, a function that takes a location `(x, y)` and outputs the temperature and pressure at that location `(T, P)`. How do we take the "derivative" of such a function? The answer is a matrix that contains all the partial derivatives, organized in a specific way. This matrix is the **Jacobian**.

#### The Motivating Problem

Imagine a simple factory. The inputs are raw materials: `x` units of steel and `y` units of plastic. The outputs are finished products: `u` cars and `v` toys.

The production functions are:
*   Cars: `u(x, y) = 2x + 0.5y` (Cars use a lot of steel, a little plastic).
*   Toys: `v(x, y) = 0.1x + 3y` (Toys use a little steel, a lot of plastic).

We can write this as a single function `F` that maps input vectors `(x, y)` to output vectors `(u, v)`.
`F(x, y) = <u(x, y), v(x, y)> = <2x + 0.5y, 0.1x + 3y>`

**The Question:** If we get one extra unit of steel (`x`), how does our production of *both* cars and toys change?

#### Build the Intuition: An Organized Table of Sensitivities

We can answer parts of this question using the partial derivatives we already know. There are four "sensitivities" we can measure:

1.  **Effect of Steel on Cars (`∂u/∂x`):** How many more cars do we get per extra unit of steel?
    `∂u/∂x = 2`. For every 1 unit of steel, we get 2 more cars.
2.  **Effect of Plastic on Cars (`∂u/∂y`):** How many more cars do we get per extra unit of plastic?
    `∂u/∂y = 0.5`. For every 1 unit of plastic, we get 0.5 more cars.
3.  **Effect of Steel on Toys (`∂v/∂x`):** How many more toys do we get per extra unit of steel?
    `∂v/∂x = 0.1`. For every 1 unit of steel, we get 0.1 more toys.
4.  **Effect of Plastic on Toys (`∂v/∂y`):** How many more toys do we get per extra unit of plastic?
    `∂v/∂y = 3`. For every 1 unit of plastic, we get 3 more toys.

The Jacobian is simply a matrix that organizes these four sensitivity values in a standardized way.

#### Formalization: The Jacobian Matrix and the Chain Rule

The Jacobian matrix `J` of a function `F(x, y) = <u(x, y), v(x, y)>` is defined as:

`J = [ [∂u/∂x, ∂u/∂y], [∂v/∂x, ∂v/∂y] ]`

**Structure:**
*   **Row 1:** All about the first output function (`u`).
*   **Row 2:** All about the second output function (`v`).
*   **Column 1:** Derivatives with respect to the first input variable (`x`).
*   **Column 2:** Derivatives with respect to the second input variable (`y`).

For our factory problem, the Jacobian matrix is constant because the production functions are linear:
`J = [ [2, 0.5], [0.1, 3] ]`

**What is it for? Linear Approximation.**
The Jacobian gives us the best linear approximation of how outputs change when inputs change. We write this using matrix-vector multiplication:

`[change in u, change in v]ᵀ ≈ J * [change in x, change in y]ᵀ`

Let's say we get a new shipment of `Δx = 10` units of steel and `Δy = 5` units of plastic. The expected change in production is:
`[Δu, Δv]ᵀ ≈ [ [2, 0.5], [0.1, 3] ] * [10, 5]ᵀ`
`Δu ≈ (2)(10) + (0.5)(5) = 20 + 2.5 = 22.5`
`Δv ≈ (0.1)(10) + (3)(5) = 1 + 15 = 16`
We expect to produce about 22.5 more cars and 16 more toys.

**The Multivariate Chain Rule**
The true power of the Jacobian is revealed when we compose functions. Suppose the total factory profit `P` depends on the number of cars `u` and toys `v` produced: `P(u, v) = 3000u + 500v`.

How does the profit change if we increase our supply of steel `x`?
This is a chain reaction: `x` affects `u` and `v`, which in turn affect `P`. The Jacobian makes this easy.

1.  **Function 1:** `F(x, y) = <u, v>` (Inputs to Products). Its derivative is `J_F`.
2.  **Function 2:** `P(u, v)` (Products to Profit). Its derivative is the gradient, `∇P = <∂P/∂u, ∂P/∂v> = <3000, 500>`. This is a 1x2 Jacobian.

The derivative of the composite function `P(F(x, y))` is the product of their derivative matrices:
`J_P(F) = J_P * J_F`
`[∂P/∂x, ∂P/∂y] = [∂P/∂u, ∂P/∂v] * [ [∂u/∂x, ∂u/∂y], [∂v/∂x, ∂v/∂y] ]`

Let's calculate `∂P/∂x`:
`∂P/∂x = (∂P/∂u)(∂u/∂x) + (∂P/∂v)(∂v/∂x)`
`∂P/∂x = (3000)(2) + (500)(0.1) = 6000 + 50 = 6050`
For every extra unit of steel, the factory's profit increases by $6050.

#### Connect to Reality: The Engine of Deep Learning

This chain rule, propagating derivatives backward through a composition of functions using matrix multiplication, is not just a mathematical curiosity. It is the exact mechanism behind **backpropagation**, the algorithm used to train virtually all modern neural networks.

A neural network is just a giant composition of functions (the "layers"). When the network makes an error, backpropagation uses the multivariate chain rule to calculate the "gradient" of the error with respect to every parameter in the network. It multiplies Jacobians backward, layer by layer, to tell each parameter how to adjust itself to reduce the error.

So while our factory example is simple, the real-world version is just a much bigger chain of these Jacobian multiplications, powering AI systems from image recognition to language translation.

## **Section 3: The Hessian (H) - The Shape of the Curve**

#### The Big Picture

In single-variable calculus, the second derivative, `f''(x)`, tells you about a curve's **concavity**.
*   If `f''(x) > 0`, the curve is concave up (shaped like a cup, holds water).
*   If `f''(x) < 0`, the curve is concave down (shaped like a frown, spills water).

In multiple dimensions, a surface's curvature is more complex. It can be concave up in one direction while being concave down in another. The perfect example is a Pringles chip or a horse's saddle. The **Hessian matrix** is the multivariable version of the second derivative. It's a tool that captures this complete, multi-directional picture of concavity.

#### The Motivating Problem

Imagine you are a hiker exploring a terrain. You find a perfectly flat spot where the slope is zero in every direction. The gradient here is zero: `∇f = <0, 0>`. But what kind of flat spot is it? There are three possibilities:
1.  The bottom of a valley (a local minimum).
2.  The top of a peak (a local maximum).
3.  A saddle point (like a mountain pass).

The gradient alone cannot tell these apart. We need to know how the surface *curves* away from the flat spot. We need a second derivative test.

#### Build the Intuition: From a Number to a Matrix

In single-variable calculus, the second derivative test is simple:
*   If `f'(a) = 0` and `f''(a) > 0`, you have a local minimum (concave up).
*   If `f'(a) = 0` and `f''(a) < 0`, you have a local maximum (concave down).

The Hessian matrix `H` does the same job for `f(x, y)`. It's a matrix because it has to store the curvature information for all directions. It contains four second-order partial derivatives:

*   `f_xx = ∂²f/∂x²`: The concavity in the pure x-direction.
*   `f_yy = ∂²f/∂y²`: The concavity in the pure y-direction.
*   `f_xy = ∂²f/∂y∂x`: How the x-slope changes as you move in the y-direction (the "twist").
*   `f_yx = ∂²f/∂x∂y`: How the y-slope changes as you move in the x-direction.

(For most functions we encounter, `f_xy = f_yx`, which makes the Hessian a symmetric matrix).

#### Formalization: The Hessian Matrix and Taylor's Theorem

The Hessian matrix `H` of a function `f(x, y)` is the matrix of its second partial derivatives.

**The Formula:**
`H = [ [f_xx, f_xy], [f_yx, f_yy] ]`

Let's compute the Hessian for our three basic shapes at the origin `(0,0)`, where `∇f = <0,0>`.

1.  **The Valley:** `f(x, y) = x² + y²` (a bowl shape)
    *   `f_x = 2x`, `f_y = 2y`
    *   `f_xx = 2`, `f_yy = 2`
    *   `f_xy = 0`, `f_yx = 0`
    *   `H = [ [2, 0], [0, 2] ]`
    The diagonal entries are positive, indicating it's concave up in both the x and y directions.

2.  **The Peak:** `f(x, y) = -x² - y²` (a dome shape)
    *   `f_x = -2x`, `f_y = -2y`
    *   `f_xx = -2`, `f_yy = -2`
    *   `f_xy = 0`, `f_yx = 0`
    *   `H = [ [-2, 0], [0, -2] ]`
    The diagonal entries are negative, indicating it's concave down in both directions.

3.  **The Saddle:** `f(x, y) = x² - y²` (a Pringles chip)
    *   `f_x = 2x`, `f_y = -2y`
    *   `f_xx = 2`, `f_yy = -2`
    *   `f_xy = 0`, `f_yx = 0`
    *   `H = [ [2, 0], [0, -2] ]`
    One diagonal entry is positive (concave up in x-dir) and one is negative (concave down in y-dir). This mix of curvatures defines a saddle point.

**The Second-Order Taylor Approximation**
The Hessian's formal role is to describe the quadratic part of a function near a point `a`.
`f(x) ≈ f(a) + ∇f(a)⋅(x-a) + ½(x-a)ᵀ H(a) (x-a)`

| Term                             | Meaning                                     | Analogy to `y = f(x)`        |
| -------------------------------- | ------------------------------------------- | ---------------------------- |
| `f(a)`                           | The starting height.                        | `f(a)`                       |
| `∇f(a)⋅(x-a)`                    | The best linear fit (the tangent plane).    | `f'(a)(x-a)`                 |
| `½(x-a)ᵀ H(a) (x-a)`             | The best **quadratic** fit (the curvature). | `½f''(a)(x-a)²`              |

This formula shows that the Gradient (`∇f`) gives the linear behavior and the Hessian (`H`) gives the quadratic behavior of a function at a point.

#### Connect to Reality: The Second Derivative Test

The Hessian is not just descriptive; it is the core of the **Second Derivative Test** for functions of multiple variables, which we will use in the next section to solve optimization problems. By calculating the Hessian at a point where the gradient is zero, we can determine if we have found a maximum, a minimum, or a saddle point. The properties of this simple 2x2 matrix will tell us everything we need to know about the local shape of our function.

## **Part 2: Optimization - The Art of Finding the Best**

---

### **Section 4: Unconstrained Optimization - Finding Peaks and Valleys**

#### The Big Picture

We have now built the necessary tools: the Gradient (`∇f`) to find flat spots and the Hessian (`H`) to determine the shape of the surface at those spots. We can now combine them to solve the central problem of unconstrained optimization: finding the maximum or minimum values of a function. This is the process of finding the highest peaks and lowest valleys on a surface.

#### The Motivating Problem

A company produces two products, X and Y. Based on market analysis, the daily profit `P` is modeled by the function:
`P(x, y) = 24x - x² + 16y - 2y²`
where `x` is the number of units of product X and `y` is the number of units of product Y.

To maximize profit, how many units of each product should the company produce?

#### The Algorithm: A Two-Step Process

The logic is a direct extension of what you learned in single-variable calculus.

| Single Variable `y=f(x)`                               | Multi-variable `z=f(x,y)`                                        |
| ------------------------------------------------------ | ------------------------------------------------------------------- |
| **Step 1:** Find critical points by solving `f'(x) = 0`. | **Step 1:** Find critical points by solving `∇f = <0, 0>`.           |
| **Step 2:** Classify using the 2nd derivative `f''(x)`. | **Step 2:** Classify using the Hessian matrix `H`.                    |

**Step 1: Find Critical Points (Where the ground is flat)**
A maximum or minimum can only occur where the surface is flat. This means the slope in every direction must be zero, which happens precisely when the gradient vector is the zero vector.
We need to solve the system of equations:
*   `∂f/∂x = 0`
*   `∂f/∂y = 0`

**Step 2: Classify with the Second Derivative Test**
Once we have a critical point `(a, b)`, we compute the Hessian `H` at that point. Then we calculate the **determinant** of the Hessian, `D = det(H) = f_xx * f_yy - (f_xy)²`.

| Condition                                            | Result                                                              |
| ---------------------------------------------------- | ------------------------------------------------------------------- |
| **1. If `D > 0` and `f_xx(a,b) > 0`**                  | Local **Minimum**. (Concave up in all directions, like a valley). |
| **2. If `D > 0` and `f_xx(a,b) < 0`**                  | Local **Maximum**. (Concave down in all directions, like a peak).  |
| **3. If `D < 0`**                                    | **Saddle Point**. (Curvature is mixed).                             |
| **4. If `D = 0`**                                    | The test is **inconclusive**. It could be anything.                 |

#### Step-by-Step: Solving the Profit Problem

**Problem:** Maximize `P(x, y) = 24x - x² + 16y - 2y²`.

**Step 1: Find the Critical Points.**
First, we compute the partial derivatives (the components of the gradient).
*   `∂P/∂x = 24 - 2x`
*   `∂P/∂y = 16 - 4y`

Now, set both to zero and solve.
*   `24 - 2x = 0  =>  2x = 24  =>  x = 12`
*   `16 - 4y = 0  =>  4y = 16  =>  y = 4`

We have found only one critical point: `(12, 4)`. This is our only candidate for a maximum.

**Step 2: Classify the Point with the Hessian.**
First, we need the four second-order partial derivatives.
*   `P_xx = ∂/∂x (24 - 2x) = -2`
*   `P_yy = ∂/∂y (16 - 4y) = -4`
*   `P_xy = ∂/∂y (24 - 2x) = 0` (Since there's no `y` in the expression)
*   `P_yx = ∂/∂x (16 - 4y) = 0`

Now, assemble the Hessian matrix. In this case, the Hessian is constant for all `(x, y)`.
`H = [ [-2, 0], [0, -4] ]`

Calculate the determinant `D`.
`D = (-2)(-4) - (0)² = 8`

Now we apply the test logic:
1.  Is `D > 0`? Yes, `8 > 0`. This means it's definitely a local max or min.
2.  Is `f_xx < 0`? Yes, `P_xx = -2 < 0`.

Since `D > 0` and `P_xx < 0`, our critical point `(12, 4)` corresponds to a **local maximum**.

#### Connect to Reality: The Answer

The mathematics is done. Now we interpret the result.
To maximize profit, the company should produce **12 units of product X** and **4 units of product Y**.

We can also calculate the maximum profit by plugging these values back into the profit function:
`P(12, 4) = 24(12) - (12)² + 16(4) - 2(4)²`
`P(12, 4) = 288 - 144 + 64 - 32 = 176`
The maximum daily profit is $176.

This two-step process—find the flat spots with the gradient, classify them with the Hessian—is the foundation of optimization. In machine learning, the "profit function" is a "loss function" that measures error, and the "products" are millions of model parameters. The goal is the same: find the parameter values that minimize the loss. The algorithm, at its core, is just a much, much larger version of what we did here.

## **Section 5: Constrained Optimization (Lagrange Multipliers) - Optimization with Rules**

#### The Big Picture

In the previous section, we found the absolute peak of a profit function. But real-world problems often have constraints. You want to maximize profit, *but* you only have a limited budget. You want to build the lightest bridge, *but* it must be able to support a certain load.

**Lagrange Multipliers** are a clever and powerful method for solving these constrained optimization problems. The technique transforms a complex constrained problem into a simpler, unconstrained system of equations.

#### The Motivating Problem

You need to build a rectangular pen. You have **24 meters** of fencing available. What are the dimensions `(x, y)` of the pen that maximize the enclosed **Area**?

Let's define our two functions:
1.  **Objective Function:** This is the function we want to maximize.
    `f(x, y) = Area = xy`
2.  **Constraint Equation:** This is the rule we must follow.
    `Perimeter = 2x + 2y = 24`. We can simplify this to `g(x, y) = x + y = 12`.

We are not looking for the maximum area in the universe; we are looking for the maximum area *on the line* `x + y = 12`.

#### Build the Intuition: The Point of Tangency

Imagine two maps laid on top of each other.
1.  A contour map of the `Area` function `f(x, y) = xy`. The contour lines are hyperbolas. Each line represents a constant area (e.g., `xy=10`, `xy=20`, `xy=30`).
2.  The `constraint line` `x + y = 12`.

You are "walking" along the constraint line and looking at the contour map of the Area function. You want to find the point on your path that touches the highest possible Area contour.

*   If the constraint line is crossing a contour line, you can move along the line to get to a higher-value contour.
*   The maximum value will occur at the exact point where the constraint line **is tangent** to a contour line. At this one magic spot, you can't increase the function's value any further without breaking the constraint.

What does this tangency mean mathematically? At the point of tangency, the two curves have the same slope. This means their **normal vectors** must point in the same direction. The normal vector to a contour line is the **gradient**.

Therefore, at the optimal point, the gradient of the objective function (`∇f`) must be parallel to the gradient of the constraint function (`∇g`).

#### Formalization: The Lagrange Multiplier Algorithm

Two vectors are parallel if one is a scalar multiple of the other. We call this scalar `λ` (lambda), the Lagrange multiplier.

**The Core Equation:**
`∇f = λ∇g`

This single vector equation, along with the original constraint, gives us a system of equations to solve.

**The Algorithm:**
1.  Identify your objective function `f(x, y)` and your constraint `g(x, y) = c`.
2.  Calculate the gradients `∇f = <f_x, f_y>` and `∇g = <g_x, g_y>`.
3.  Set up the system of equations:
    *   `f_x = λ * g_x`
    *   `f_y = λ * g_y`
    *   `g(x, y) = c` (The original constraint)
4.  Solve this system for `x`, `y`, and `λ`. The `(x, y)` solution is your candidate for a max or min.

#### Step-by-Step: Solving the Fence Problem

**1. Identify Functions:**
*   Objective: `f(x, y) = xy`
*   Constraint: `g(x, y) = x + y = 12`

**2. Calculate Gradients:**
*   `∇f = <∂/∂x(xy), ∂/∂y(xy)> = <y, x>`
*   `∇g = <∂/∂x(x+y), ∂/∂y(x+y)> = <1, 1>`

**3. Set up the System:**
The equation `∇f = λ∇g` becomes `<y, x> = λ<1, 1>`.
This gives us three equations:
1.  `y = λ * 1  =>  y = λ`
2.  `x = λ * 1  =>  x = λ`
3.  `x + y = 12` (the constraint)

**4. Solve the System:**
This is a simple one. From equations (1) and (2), we see that `x = y`.
Now substitute this into equation (3):
`x + (x) = 12`
`2x = 12`
`x = 6`

Since `x = y`, we also have `y = 6`. (And `λ = 6`, though we don't always need its value).

The optimal dimensions are **6 meters by 6 meters** (a square). The maximum area is `f(6, 6) = 6 * 6 = 36` square meters.

#### Connect to Reality: Economics and Engineering

This method is fundamental in many fields.
*   **Economics:** A company wants to maximize production `P(k, l)` (where `k` is capital, `l` is labor) subject to a budget constraint `B(k, l) = c`. Solving `∇P = λ∇B` finds the optimal allocation of resources. The value of `λ` itself has a meaning: it's the "shadow price," telling the company how much their production would increase for one extra dollar of budget.
*   **Engineering:** Find the strongest shape for a beam (`f` = strength) using a fixed amount of material (`g` = weight = constant).
*   **Physics:** Systems tend to settle in the lowest energy state (`f` = energy) subject to physical constraints (`g` = conservation laws).

Lagrange Multipliers provide a systematic, almost mechanical way to solve a vast range of real-world optimization problems where resources are limited.

## **Section 6: Double and Triple Integrals - Calculating Volume and Mass**

#### The Big Picture

In single-variable calculus, the definite integral `∫ₐᵇ f(x)dx` represents the **area** under a curve. We found this by slicing the interval `[a, b]` into tiny pieces `dx`, multiplying by the height `f(x)`, and summing them up.

A **double integral** `∬_R f(x,y) dA` extends this idea to 3D. We are now summing up values over a 2D region `R` in the `xy`-plane.
*   If `f(x, y) = 1`, the double integral gives the **Area** of the region `R`.
*   If `f(x, y)` represents a height, the double integral gives the **Volume** under the surface `z = f(x, y)`.
*   If `f(x, y)` represents density, the double integral gives the total **Mass** of the region `R`.

The core idea is the same: slice, multiply, sum.

#### The Motivating Problem

Find the volume of the solid region under the parabolic surface `z = 4 - x² - y²` and above the square region `R` in the `xy`-plane defined by `0 ≤ x ≤ 1` and `0 ≤ y ≤ 1`.

#### Build the Intuition: From Slices to Columns

To find the volume, we can slice the 3D solid. But instead of slicing it into 2D sheets, we will chop it into infinitely many tall, skinny rectangular columns.

1.  **Chop the Base:** Take the square region `R` in the `xy`-plane and divide it into a grid of tiny rectangular patches. The area of each tiny patch is `dA = dx dy`.
2.  **Find the Height:** Above each tiny patch, the height of the solid is given by the function `z = f(x, y) = 4 - x² - y²`.
3.  **Calculate Column Volume:** The volume of one skinny column is `dV = height × base_area = f(x, y) * dA`.
4.  **Sum Them Up:** The double integral is the symbol for summing up the volumes of all these columns over the entire region `R`.
    `Volume = ∬_R (4 - x² - y²) dA`

#### Formalization: Iterated Integrals

Calculating a double integral is done by calculating two single integrals, one after another. This is called an **iterated integral**. The key is to correctly set up the limits of integration.

The expression `∬_R f(x,y) dA` becomes either:
`∫_c^d [ ∫_a^b f(x,y) dx ] dy`  or  `∫_a^b [ ∫_c^d f(x,y) dy ] dx`

**The Rule:** You work from the inside out.
For `∫_c^d ∫_a^b f(x,y) dx dy`:
1.  **Inner Integral:** First, integrate with respect to `x`, treating `y` as a constant. The bounds `a` and `b` are for `x`.
2.  **Outer Integral:** Then, integrate the result of the inner integral with respect to `y`. The bounds `c` and `d` are for `y`.

#### Step-by-Step: Solving the Volume Problem

**Problem:** Find the volume `V = ∬_R (4 - x² - y²) dA` where `R` is the square `0 ≤ x ≤ 1`, `0 ≤ y ≤ 1`.

**1. Set up the Iterated Integral:**
Since the bounds are constant, the order doesn't matter. Let's choose `dx dy`.
`V = ∫₀¹ ∫₀¹ (4 - x² - y²) dx dy`

**2. Solve the Inner Integral (with respect to x):**
We treat `y` as a constant.
`∫₀¹ (4 - x² - y²) dx`
`= [4x - (x³/3) - y²x] from x=0 to x=1`
`= (4(1) - (1³/3) - y²(1)) - (0)`
`= 4 - 1/3 - y² = 11/3 - y²`
The result of the inner integral is a function of `y`, which is what we expect.

**3. Solve the Outer Integral (with respect to y):**
Now we integrate our result from the previous step.
`V = ∫₀¹ (11/3 - y²) dy`
`= [11/3 * y - y³/3] from y=0 to y=1`
`= (11/3 * 1 - 1³/3) - (0)`
`= 11/3 - 1/3 = 10/3`

The volume under the surface over the unit square is exactly **10/3**.

#### A More Complex Example: Variable Bounds

**Problem:** Find the mass of a triangular plate with vertices at `(0,0)`, `(2,0)`, and `(2,1)`. The density of the plate is given by `ρ(x, y) = xy`.
`Mass = ∬_R xy dA`

**The Challenge:** The bounds are not constant. The region `R` is a triangle. We need to describe this region with inequalities.
*   `x` goes from `0` to `2`.
*   For a given `x`, `y` goes from the bottom edge (`y=0`) to the top slanted edge. The line connecting `(0,0)` to `(2,1)` is `y = x/2`.

So, the bounds are: `0 ≤ x ≤ 2` and `0 ≤ y ≤ x/2`.

**Setting up the integral:** The variable bound (`y = x/2`) *must* go on the inside.
`Mass = ∫₀² ∫₀^(x/2) xy dy dx`

1.  **Inner integral (w.r.t. y):**
    `∫₀^(x/2) xy dy = x [y²/2] from y=0 to y=x/2 = x * ((x/2)²/2) = x * (x²/8) = x³/8`

2.  **Outer integral (w.r.t. x):**
    `Mass = ∫₀² (x³/8) dx = [x⁴/32] from 0 to 2 = 2⁴/32 = 16/32 = 1/2`

The total mass of the plate is **1/2**.

#### Connect to Reality: From Volume to Probability

Double and triple integrals are used everywhere:
*   **Physics & Engineering:** Calculating the mass, center of mass, and moment of inertia of objects with non-uniform density.
*   **Computer Graphics:** Calculating the total light hitting a surface by integrating brightness over an area.
*   **Probability & Statistics:** A "joint probability density function" `f(x, y)` describes the likelihood of two random variables. The probability that `(x, y)` falls into a region `R` is found by computing `∬_R f(x,y) dA`. The total probability must integrate to 1.

## **Section 7: Change of Variables (Jacobians Again) - Finding the Right Perspective**

#### The Big Picture

In the last section, we saw that setting up the bounds of integration can be difficult if the region `R` is not a simple rectangle. Integrating over circular, spherical, or other curved regions in Cartesian coordinates (`x, y, z`) often leads to messy integrals involving square roots.

The **Change of Variables** technique allows us to switch to a more natural coordinate system (like polar or spherical) that makes the region and the integral simpler. But this switch comes at a cost: we distort the area (or volume) of our tiny integration patches. The **Jacobian determinant** is the precise "fudge factor" we need to correct for this distortion.

#### The Motivating Problem

Find the volume of the solid under the parabolic surface `z = 9 - x² - y²` and above the `xy`-plane.

The base of this solid is a circle in the `xy`-plane. To see this, we find where the surface intersects the plane (`z=0`):
`0 = 9 - x² - y²  =>  x² + y² = 9`
This is a circle of radius 3.

Let's try to set this up as a double integral in Cartesian coordinates:
`Volume = ∫₋₃³ ∫₋√₍₉₋ₓ²₎^√₍₉₋ₓ²₎ (9 - x² - y²) dy dx`

This is a terrible integral. The bounds involve square roots, and the integrand will become complicated. The problem has circular symmetry, so Cartesian coordinates are the wrong tool for the job. Polar coordinates will be much easier.

#### Build the Intuition: The Area Distortion Factor

Let's switch from Cartesian `(x, y)` to Polar `(r, θ)`.
*   `x = r cos(θ)`
*   `y = r sin(θ)`

In the Cartesian world, our area element `dA` is a simple rectangle, `dA = dx dy`.
In the Polar world, what is the area element? If we take a small step `dr` and a small turn `dθ`, the patch we sweep out is not a rectangle. It is a small, curved sector.

The area of this polar patch is approximately `(r dθ) * dr = r dr dθ`. Notice the extra `r`. A patch far from the origin (large `r`) has a larger area than a patch near the origin (small `r`) for the same `dr` and `dθ`.

This extra factor of `r` is the "fudge factor" that corrects for the distortion of area when moving from a Cartesian grid to a polar grid. This factor is, in fact, the determinant of the Jacobian matrix for the coordinate transformation.

#### Formalization: The Change of Variables Formula

If we change variables from `(x, y)` to `(u, v)`, the formula is:
`∬_R f(x,y) dx dy = ∬_S f(u,v) |det(J)| du dv`

Where `J` is the Jacobian of the transformation from `(u,v)` to `(x,y)`.
`J = [ [∂x/∂u, ∂x/∂v], [∂y/∂u, ∂y/∂v] ]`

**Let's calculate it for Polar Coordinates:** `u=r`, `v=θ`.
`x = r cos(θ)` and `y = r sin(θ)`.
*   `∂x/∂r = cos(θ)`
*   `∂x/∂θ = -r sin(θ)`
*   `∂y/∂r = sin(θ)`
*   `∂y/∂θ = r cos(θ)`

The Jacobian matrix is: `J = [ [cos(θ), -r sin(θ)], [sin(θ), r cos(θ)] ]`
The determinant is:
`det(J) = (cos(θ))(r cos(θ)) - (-r sin(θ))(sin(θ))`
`= r cos²(θ) + r sin²(θ)`
`= r (cos²(θ) + sin²(θ)) = r(1) = r`
The absolute value `|det(J)| = r`.

This confirms our intuition. The conversion factor is `r`.
**`dx dy = r dr dθ`**

#### Step-by-Step: Solving the Volume Problem (The Easy Way)

**Problem:** Find the volume under `z = 9 - x² - y²` over the circle `x² + y² ≤ 9`.

**1. Convert Everything to Polar Coordinates:**
*   **The Region:** The circle `x² + y² ≤ 9` is incredibly simple in polar. It's just `0 ≤ r ≤ 3` and `0 ≤ θ ≤ 2π`. Our messy square-root bounds have become constant bounds.
*   **The Function:** The integrand `9 - x² - y²` becomes `9 - (x² + y²) = 9 - r²`.
*   **The Area Element:** `dA = dx dy` becomes `r dr dθ`.

**2. Set up the New Integral:**
`Volume = ∫₀²π ∫₀³ (9 - r²) * r dr dθ`
`Volume = ∫₀²π ∫₀³ (9r - r³) dr dθ`

**3. Solve the Integral:**
This is much easier than the Cartesian version.
*   **Inner Integral (w.r.t. r):**
    `∫₀³ (9r - r³) dr = [9r²/2 - r⁴/4] from r=0 to r=3`
    `= (9(3)²/2 - 3⁴/4) - (0) = 81/2 - 81/4 = 81/4`

*   **Outer Integral (w.r.t. θ):**
    `∫₀²π (81/4) dθ = (81/4) [θ] from 0 to 2π`
    `= (81/4) * (2π - 0) = 81π / 2`

The volume is **81π / 2**. This would have been a nightmare to solve in Cartesian coordinates.

#### Connect to Reality: Symmetry Exploits

This technique is essential in physics and engineering. Many real-world problems have symmetry.
*   **Cylindrical Coordinates `(r, θ, z)`:** Used for problems involving pipes, cylinders, or wires. The volume element is `dV = r dz dr dθ`.
*   **Spherical Coordinates `(ρ, θ, φ)`:** Used for problems involving planets, stars, or atoms. The volume element is `dV = ρ² sin(φ) dρ dθ dφ`.

Whenever you see `x² + y²` in an integral, think polar. Whenever a problem involves spheres or gravity, think spherical coordinates. The Jacobian gives us the key to unlock these simpler perspectives, turning difficult problems into manageable ones.


## **Section 8: Conclusion**

We have journeyed from the flat world of `y = f(x)` into the rich, multi-dimensional space of `z = f(x, y)` and beyond. We discovered that the core ideas of calculus—derivatives and integrals—do not break, but rather expand in beautiful and powerful ways.

The derivative is no longer just a slope; it is a **gradient vector (`∇f`)** that points the way up the steepest hill. The second derivative is no longer just a number for concavity; it is a **Hessian matrix (`H`)** that describes the full curvature of a surface—a peak, a valley, or a saddle. Together, these tools gave us a robust algorithm for optimization.

The integral is no longer just the area under a curve; it is the **volume under a surface (`∬ f dA`)** or the mass of a complex object. We learned that changing our perspective—switching coordinate systems—can turn an impossible integral into a simple one, with the **Jacobian** acting as the universal translator.

You started this tutorial an hour ago. The concepts and symbols below are no longer abstract. They are a powerful toolkit for solving problems in engineering, physics, computer science, and economics.

**The Toolbox You Have Mastered**

| Technique/Concept              | Notation / Formula                                     | What It Solves                                                                |
| ------------------------------ | ------------------------------------------------------ | ----------------------------------------------------------------------------- |
| **The Gradient**               | `∇f = <∂f/∂x, ∂f/∂y>`                                  | Finds the direction of steepest ascent on a surface.                            |
| **The Jacobian**               | `J` (matrix of partials)                               | Translates how changes in inputs affect outputs in multiple dimensions.         |
| **The Hessian**                | `H` (matrix of 2nd partials)                           | Determines the shape of a surface: peak, valley, or saddle.                     |
| **Unconstrained Optimization** | `∇f = 0`, then test `H`                                | Finds and classifies the maximums and minimums of a function.                 |
| **Lagrange Multipliers**       | `∇f = λ∇g`                                             | Finds the maximum/minimum of a function subject to a constraint.              |
| **Double Integrals**           | `∬ f(x,y) dA`                                          | Calculates the volume under a surface or the mass of a variable-density plate. |
| **Change of Variables**        | `dx dy = |det(J)| du dv`                                | Simplifies integrals by switching to a more natural coordinate system (like polar). |

Multivariable calculus now clicks.