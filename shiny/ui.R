library(shiny)

shinyUI(fluidPage(
  titlePanel(h3("Avon Metastatic Breast Cancer Abstract Analysis")),

  sidebarLayout(position = "right",
    sidebarPanel( 
      textInput("abstractIndex", label = h3("Abstract Index"), value = "Enter abstract index, 0-99"),
      submitButton("Submit")
      ),
    mainPanel(
      h5("Award Title"),
      textOutput("AwardTitle"),
      h5("Abstract"), 
      textOutput("TechAbstract"),
      h5("Pathway"),
      textOutput("Pathway"),
      h5("Pathway Group"),
      textOutput("PathwayGroup"),
      h5("Molecular Target"),
      textOutput("MolecularTarget"),
      h5("Molecular Target Group"),
      textOutput("MolecularTargetGroup")
      )
  )
))
