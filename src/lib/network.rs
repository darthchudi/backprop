use rand::Rng;
use crate::value::Value;

// Weight represents the default weights type of float64 numbers
// wrapped within the Value type.
type Weight = Value<f64>;
type Bias = Weight;

pub struct Neuron {
    weights: Vec<Weight>,
    bias: Bias,
}

// Neuron represents a single neuron with a given weight and bias value
impl Neuron {
    fn new(inputs: u64) -> Neuron {
        let mut weights_rng = rand::thread_rng();
        let mut bias_rng = rand::thread_rng();

        let mut weights: Vec<Value<f64>> = Vec::with_capacity(inputs as usize);
        for _ in 0..inputs {
            let raw_weight = weights_rng.gen_range(-1.0..=1.0);
           let weight = Value::new(raw_weight);

            weights.push(weight);
        }

        let raw_bias = bias_rng.gen_range(-0.01..=0.01);
        let bias = Value::new(raw_bias);

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
        // Compute the weighted sum of inputs for the neuron.
        let weighted_sum = self.weights.
            iter().
            zip(x).
            map(|(w, input)| {
                let input_dim = &Value::new_from_ref(input);
                
                w * input_dim
            }).
            fold(Value::new(0.0), |acc, item| {
                acc + item
            });
        
        let weight_and_bias = &weighted_sum + &self.bias;

        let activation = Neuron::relu(weight_and_bias.get_data());

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

#[cfg(test)]
mod tests {
    use crate::{network};

    #[test]
    fn simple_network() {
        let network = network::Network{
            layers: vec![
                network::Layer::new(3, 4),
                network::Layer::new(4, 5),
                network::Layer::new(5, 100),
                network::Layer::new(100, 1),
            ],
        };

        let inputs = vec![0.1, 0.2, 0.3];

        let output = network.forward(&inputs);
        
        println!("{:?}", output);
    }
}
