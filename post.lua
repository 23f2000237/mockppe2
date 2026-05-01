wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"

request = function()
    local body = [[
    {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    ]]

    return wrk.format(nil, "/predict/", nil, body)
end
