

def test_model(model,X_test,Y_test):
    scores=model.evaluate(X_test,Y_test,verbose=1)
    print(model.metrics_names[1],scores[1]*100)
    return model.metrics_names[1],scores[1]*100
