use rand::Rng;

pub struct Neuron {
    weights: Vec<f64>,
    bias: f64
}

// Neuron represents a single neuron with a given weight and bias value
impl Neuron {
    fn new(inputs: u64) -> Neuron {
        let mut weights_rng = rand::thread_rng();
        let mut bias_rng = rand::thread_rng();

        let mut weights: Vec<f64> = Vec::with_capacity(inputs as usize);
        for _ in 0..inputs {
            let weight = weights_rng.gen_range(-1.0..=1.0);
            weights.push(weight);
        }

        let bias = bias_rng.gen_range(-0.01..=0.01);

        Neuron {
            weights,
            bias
        }
    }

    fn relu(x: f64) -> f64 {
        x.max(0.0)
    }

    // Performs the forward pass on a given input and returns the activation
    fn forward(&self, x: &Vec<f64>) -> f64 {
        let mut weight_sum = 0.0;

        if x.len() != self.weights.len() {
            panic!("{}", format!("{} input dimensions not compatible with {} weight dimensions", x.len(), self.weights.len()));
        }

        // Compute the weighted sum of inputs for the neuron.
        for (index, input_val) in x.iter().enumerate() {
            let computed_val = self.weights[index]  * input_val;  // Wi * Xi
            weight_sum += computed_val; // Sum(WnXn)
        }

        let weight_and_bias = weight_sum + self.bias;

        let activation = Neuron::relu(weight_and_bias);

        activation
    }
}

// Layer consists of a set of neurons which receive inputs
pub struct Layer {
    neurons: Vec<Neuron>
}

impl Layer {
    pub fn new(num_inputs: u64, num_outputs: u64) -> Layer{
        let mut neurons = Vec::with_capacity(num_outputs as usize);

        for _ in 0..num_outputs {
            let neuron = Neuron::new(num_inputs);
            neurons.push(neuron);
        }

        Layer{neurons}
    }

    fn forward(&self, inputs: &Vec<f64>) -> Vec<f64> {
        let mut outputs = Vec::with_capacity(self.neurons.len());

        for neuron in &self.neurons{
            let neuron_result = neuron.forward(&inputs);
            outputs.push(neuron_result);
        }

        outputs
    }
}

pub struct Network {
   pub layers: Vec<Layer>
}

impl Network {
    pub fn forward(&self, inputs: &Vec<f64>) -> Vec<f64> {
        let mut result = Vec::new();

        for (index, layer) in self.layers.iter().enumerate(){
            if index == 0 {
                // The first layer receives the inputs directly
                result = layer.forward(&inputs);
                continue
            }

            // Subsequent layers will receive the previous layers output
            result = layer.forward(&result)
        }

        result
    }
}
