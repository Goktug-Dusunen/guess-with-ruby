require 'numpy'
require 'keras'
require 'matplotlib'

def predict_future_values(file_name, num_predictions)
  # Read the data from the file
  data = []
  File.open(file_name, "r") do |f|
    f.each_line do |line|
      data << line.to_f
    end
  end

  # Normalize the data
  data = (data - data.mean) / data.stddev

  # Build the model
  model = Keras::Sequential.new
  model.add(Keras::Layers::Dense.new(units: 128, activation: 'relu', input_shape: [1]))
  model.add(Keras::Layers::Dense.new(units: 128, activation: 'relu'))
  model.add(Keras::Layers::Dense.new(units: 128, activation: 'relu'))
  model.add(Keras::Layers::Dense.new(units: 1))
  model.compile(optimizer: 'adam', loss: 'mean_squared_error')

  # Train the model
  x_train = Numpy.array(data[0..-2]).reshape(-1, 1)
  y_train = Numpy.array(data[1..-1]).reshape(-1, 1)
  model.fit(x_train, y_train, epochs: 1000, verbose: 0)

  # Use the model to generate predictions
  predictions = []
  x_test = Numpy.array(data[-1]).reshape(1, 1)
  num_predictions.times do
    y_test = model.predict(x_test)
    predictions << y_test[0][0] * data.stddev + data.mean
    x_test = y_test
  end

  return predictions
end

def plot_predictions(data, predictions)
  x = (0...data.length).to_a
  y = data
  x_pred = (data.length...data.length + predictions.length).to_a
  y_pred = predictions

  Matplotlib::Pyplot.plot(x, y, 'bo', label: 'Historical Data')
  Matplotlib::Pyplot.plot(x_pred, y_pred, 'ro', label: 'Predictions')
  Matplotlib::Pyplot.legend()
  Matplotlib::Pyplot.show()
end

# Example usage
data = []
File.open("member.txt", "r") do |f|
  f.each_line do |line|
    data << line.to_f
  end
end

predictions = predict_future_values("member.txt", 8)
puts predictions

plot_predictions(data, predictions)
