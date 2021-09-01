import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State


import Bert_Comment
import Bert_trained

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Political Sentiment Analysis", style = {'text-align': 'center'}),
        "Please Enter Subreddit: ",
        dcc.Input(id='sub_input',
                  value='Tories',
                  type='text',
                  ),
        html.Br(),
        html.Br(),
        "Number of Submissions: ",
        dcc.Input(id='num_input',
                  value=10,
                  type='number',
                  min=10,
                  ),
        html.Button(id='submit-button-state', n_clicks = 0, children="Submit"),

    html.Br(),
    html.Br(),
    html.Div('Result :',
             id='my-output'),

    'Please Enter a Comment: ',
             dcc.Textarea(id="comment_input"),
              html.Button(id='comment_submit', n_clicks = 0, children='Submit'),
              html.Div('Results: ',id='comment_output')


])

@app.callback(
    Output(component_id='my-output', component_property='children'),
    Input(component_id='submit-button-state', component_property='n_clicks'),
    State(component_id='sub_input',component_property= 'value'),
    State(component_id='num_input', component_property='value'),
    prevent_initial_call = True
)
def reddit_analysis(n_clicks,sub_input, num_input):
    print(sub_input)
    print(type(sub_input))
    test = Bert_trained.sentiment(sub_input, num_input)
    return '{}'.format(test)

@app.callback(
    Output('comment_output','children'),
    Input('comment_submit', 'n_clicks'),
    State('comment_input', 'value'),
    prevent_intial_call = True
)
def text_area_output (n_clicks, comment_input):
    if n_clicks > 0:
        print(comment_input)
        test = Bert_Comment.sentiment(comment_input)
        return 'Your comment is: \n{}'.format(test)





if __name__ == '__main__':
    app.run_server(debug=True)