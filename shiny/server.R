library(shiny)
library(synapseClient)

# TODO rather than logging in with a service account, have the user provide their credentials below
synapseLogin()

shinyServer(function(input, output) {
  # TODO have user log in here, e.g. synapseLogin(input$user, input$password, rememberMe=F)

  # TODO rather than doing a single query, select the requested column for the given index, on the fly
  table<-synTableQuery("SELECT * FROM syn5479989 LIMIT 100")

   output$TechAbstract<-renderText(
    {
      rowIndex<-grep(sprintf("^%s_", input$abstractIndex), rownames(table@values))
      if (length(rowIndex)!=1) {
        "Error:  No unique matching record"
      } else {
        table@values[rowIndex, "TechAbstract"]
      }
    }
  )
      
  output$AwardTitle<-renderText(
    {
      rowIndex<-grep(sprintf("^%s_", input$abstractIndex), rownames(table@values))
      if (length(rowIndex)!=1) {
        "Error:  No unique matching record"
      } else {
        table@values[rowIndex, "AwardTitle"]
      }
    }
  )

      
  output$Pathway<-renderText(
    {
      rowIndex<-grep(sprintf("^%s_", input$abstractIndex), rownames(table@values))
      if (length(rowIndex)!=1) {
        "Error:  No unique matching record"
      } else {
        table@values[rowIndex, "Pathway"]
      }
    }
  )

      
  output$PathwayGroup<-renderText(
    {
      rowIndex<-grep(sprintf("^%s_", input$abstractIndex), rownames(table@values))
      if (length(rowIndex)!=1) {
        "Error:  No unique matching record"
      } else {
        table@values[rowIndex, "Pathway_Group"]
      }
    }
  )

      
  output$MolecularTarget<-renderText(
    {
      rowIndex<-grep(sprintf("^%s_", input$abstractIndex), rownames(table@values))
      if (length(rowIndex)!=1) {
        "Error:  No unique matching record"
      } else {
        table@values[rowIndex, "Molecular_Target"]
      }
    }
  )
      
  output$MolecularTargetGroup<-renderText(
    {
      rowIndex<-grep(sprintf("^%s_", input$abstractIndex), rownames(table@values))
      if (length(rowIndex)!=1) {
        "Error:  No unique matching record"
      } else {
        table@values[rowIndex, "Molecular_Target_Group"]
      }
    }
  )


})
