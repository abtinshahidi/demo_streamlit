import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

st.title("Exploring linear regression")
st.markdown("""
In this interactive notebook, we are going to sample from a linear distribution
and we add a noise to these data points and will find the posterior Distribution
over the models parameters.
""")

@st.cache()
def generate_linear_data(N, a=1, b=0, sigma=1):
    x = np.random.random(N) * 10
    return np.c_[x, a * x + b + np.random.normal(0, scale=sigma, size=x.shape[0])]

N  = st.slider("Number of data points", 0, 1, 50, 1)
sigma = st.slider("Noise Level", 0.01, 1., 2., step=0.01)


st.markdown("""
In the following, we choose the true values for **slope** ($a$) and **intercept** ($b$).
""")


st.subheader("Selecting true linear function")
st.latex("ax + b")

a = st.slider("a", 0.01, 1., 1.5, step=0.01)
b = st.slider("b", 0.01, -1., 1., step=0.01)

x_data, y_data = generate_linear_data(N, a=a, b=b, sigma=sigma).T



########### Figure ########
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1, 1, 1)
ax.plot(x_data, y_data, ".", markersize=18)
ax.set_ylabel(r"y", fontsize=25)
ax.set_xlabel(r"x", fontsize=25)
ax.tick_params(labelsize=18)
st.subheader("The distribution of $N={}$ points".format(N))
st.pyplot(fig)






def least_squared(data, prediction):
    diff = (data-prediction)**2
    return sum(diff)

@st.cache()
def find_posterior():

    num = 200

    α_span = np.tan(np.linspace(0, np.pi/3, num))
    β_span = np.linspace(-3, 3, num)

    _matrix_err_ = np.zeros((num, num))

    for i in range(num):
        for j in range(num):
            prediction = α_span[i] * x_data + β_span[j]
            _matrix_err_[i][j] = least_squared(y_data, prediction)



    ij_min = np.where(_matrix_err_ == _matrix_err_.min())

    our_best_estimate = (α_span[ij_min[0]], β_span[ij_min[1]])

    XX, YY = np.meshgrid(α_span, β_span)
    _X_ = np.c_[XX.flatten(), YY.flatten()]

    marginal_0 = np.exp(-_matrix_err_).sum(axis=0)
    marginal_1 = np.exp(-_matrix_err_).sum(axis=1)
    return marginal_0, marginal_1, our_best_estimate, α_span, β_span, _matrix_err_, XX, YY

marginal_0, marginal_1, our_best_estimate, α_span, β_span, _matrix_err_,  XX, YY = find_posterior()


@st.cache()
def find_confidence(probability, x, confidence=0.95, density=True):
    if density:
        probability = probability / trapz(probability, x)
    else:
        probability = probability / probability.sum()

    most_probable = probability[probability==probability.max()]
    k = 0
    integrate = 0
    while integrate < confidence:
        k += 1
        selection = (probability >= (1 - 0.001 * k) * most_probable)
        integrate = trapz(probability[selection], x[selection])

    x_lower = x[selection][0]
    x_upper = x[selection][-1]
    print(integrate, x[probability==most_probable])

    return [x_lower, x_upper]

confidence = st.slider("Confidence Interval", 0.01, 0.95, 0.99, step=0.01)



[x0, x1] = find_confidence(marginal_0, β_span, confidence=confidence)
[y0, y1] = find_confidence(marginal_1, α_span, confidence=confidence)




# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
spacing = 0.005


selection_1 = [(α_span >= y0) * (α_span <= y1)]
selection_0 = [(β_span >= x0) * (β_span <= x1)]


rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.2, height]

# start with a square Figure
fig1 = plt.figure(figsize=(8, 8))

ax = fig1.add_axes(rect_scatter)
ax.contourf(XX, YY, np.log10(_matrix_err_.T), levels=20, cmap="Spectral")
ax.scatter(our_best_estimate[0], our_best_estimate[1])
ax.vlines(our_best_estimate[0], *ax.get_ylim(), linestyles="dashed")
ax.hlines(our_best_estimate[1], *ax.get_xlim(), linestyles="dashed")

ax.vlines(α_span[selection_1][0], *ax.get_ylim(), linestyles="dotted")
ax.vlines(α_span[selection_1][-1], *ax.get_ylim(), linestyles="dotted")


ax.hlines(β_span[selection_0][0], *ax.get_xlim(), linestyles="dotted")
ax.hlines(β_span[selection_0][-1], *ax.get_xlim(), linestyles="dotted")

ax.scatter(a, b, c="r", s=120, marker="*")


ax.tick_params(labelsize=18)
ax.set_xlabel(r"$a$", fontsize=40)
ax.set_ylabel(r"$b$", fontsize=40)


ax_histx = fig1.add_axes(rect_histx, sharex=ax)
ax_histx.plot(α_span, marginal_1)
ax_histx.fill_between(α_span[selection_1], marginal_1[selection_1], color="c", alpha=0.5)
ax_histx.vlines(α_span[marginal_1==marginal_1.max()], *ax_histx.get_ylim(), linestyles="dashed")
ax_histx.axis("off")



ax_histy = fig1.add_axes(rect_histy, sharey=ax)
ax_histy.plot(marginal_0, β_span)
ax_histy.fill_betweenx( β_span[selection_0],marginal_0[selection_0], color="c", alpha=0.5)
ax_histy.hlines(β_span[marginal_0==marginal_0.max()], *ax_histy.get_xlim(), linestyles="dashed")


ax.text(our_best_estimate[0] - 0.8, 3.75, "$a={}$".format(round(float(our_best_estimate[0]), 3)), fontsize=32)
ax.text(our_best_estimate[0] + 0.2, 3.8, "$ p(a \in [{}, {}]) = {}$".format(round(float(y0), 3), round(float(y1), 3), confidence), fontsize=22)


ax.text(2, our_best_estimate[1] + 1, "$b={}$".format(round(float(our_best_estimate[1]), 3)),  fontsize=32)
ax.text(2, our_best_estimate[1] - 1, "$ p(b \in [{}, {}]) = {}$".format(round(float(x0), 3), round(float(x1), 3), confidence), fontsize=22)


ax_histy.axis("off")

fig1.suptitle(r"Joint Probability Distribution $p(a, b)$".format(confidence), fontsize=30, y=1.02)
st.header("Joint probability distribution over model parameters")

st.pyplot(fig1)
st.markdown("True value of parameters $a = {}, b = {}$".format(a, b))
