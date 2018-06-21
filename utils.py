def log_training_loss(engine, data_loader, log_interval):
    iter = (engine.state.iteration - 1) % len(data_loader) + 1
    if iter % log_interval == 0:
        print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
              "".format(engine.state.epoch, iter, len(data_loader), engine.state.output))

def log_training_results(engine, evaluator, data_loader, metrics, is_train=True):
    evaluator.run(data_loader)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_loss = metrics['loss']
    results_text = "Train" if is_train else "Validation"
    print("{} Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
          .format(results_text, engine.state.epoch, avg_accuracy, avg_loss))