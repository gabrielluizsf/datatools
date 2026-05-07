package datatools

// BellmanEquation calculates the target Q-learning value using the Bellman Equation.
//
// Formula:
//
// Q(s, a) = r + γ * max(Q(s', a'))
//
// Where:
// 	- r = immediate reward
// 	- γ = discount factor (gamma)
// 	- max(Q(s', a')) = highest Q value of the next state
//
// Parameters:
// 	- reward: immediate reward received
// 	- gamma: discount factor between 0 and 1
// 	- nextMaxQ: highest estimated Q value for the next state
//
// Return:
// 	- updated value of the Bellman equation
func BellmanEquation(
	reward float64,
	gamma float64,
	nextMaxQ float64,
) float64 {
	return reward + (gamma * nextMaxQ)
}
