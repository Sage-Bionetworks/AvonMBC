function createMenu(file) {
    $.getJSON(file,function(dat) {
        var select = document.getElementById("menuselect");
        for (i=1; i< dat['Number of Grants']+1;i++) {
            select.options[select.options.length] = new Option(i+'.json', i+'.json');
        }
    })
}

function mbc_dash(file) {
    var width = 1300;
        height = 550;

    var title = d3.select("body")
            .append("div")
            .attr("width",width)
            .attr("height",height)
            .attr("class","title")
            .style('font-family','Helvetica')
            .style('font-size','20px')
            .attr("id","title")

    var abstract = d3.select("body")
            .append('div')
            .attr('width',width)
            .attr('height',height)
            .attr('class','abstract')
            .style('font-family','Helvetica')
            .attr("id","abstract")
            .style('font-size','10px')

    var info = d3.select("body")
            .append('div')
            .attr('width',width)
            .attr('height',height)
            .attr('class','info')
            .style('font-family','Helvetica')
            .attr("id","info")
            .style('font-size','10px') 	

    //Pulldown menu
    var menu = d3.select('#menuselect').on("change", function() {
        var newFile = d3.select('#menuselect').property('value')
        title.selectAll('*').remove()
        abstract.selectAll('*').remove()
        info.selectAll('*').remove()
        display(newFile)

    });

    
    function display(file) {
    	
    	$.getJSON(file,function(dat){

    		var title = '<p>'+'<b>Title</b>: '+dat['AwardTitle'] + '</p>'
            var abstract = '<p>'+'<b> Abstract</b>: '+dat['TechAbstract']+ '</p>'
	    	var information = '<p>'+'<b>&nbspAuthor</b>: '+dat['PILastName']+', '+ dat['PIFirstName']+ '</p>'
            information += '<p>'+'<b>&nbspCity</b>: '+dat['City']+ '</p>'
            information += '<p>'+'<b>&nbspCountry</b>: '+dat['Country']+ '</p>'
            information += '<p>'+'<b>&nbspInstitution</b>: '+dat['Institution']+ '</p>'
            information += '<p>'+'<b>&nbspTarget</b>: '+dat['Molecular Target']+'</p>'
            information += '<p>'+'<b>&nbspTarget (Group)</b>: '+dat['Molecular Target (Group)']+ '</p>'
            information += '<p>'+'<b>&nbspPathway</b>: '+dat['Pathway']+ '</p>'
            information += '<p>'+'<b>&nbspPathway (Group)</b>: '+dat['Pathway (Group)']+ '</p>'
            information += '<p>'+'<b>&nbspMetastasis (Y/N)</b>: '+dat['Metastasis Y/N']+'</p>'
            information += '<p>'+'<b>&nbspMetastasis Stage</b>: '+dat['*Metastasis stage']+ '</p>'
            information += '<p>'+'<b>&nbspdb_mt</b>: '+dat['db_mt']+ '</p>'
            information += '<p>'+'<b>&nbspdb_pw</b>: '+dat['db_pw']+ '</p>'
            information += '<p>'+'<b>&nbspdb_pwgene</b>: '+dat['db_pwgene']+ '</p>'
            information += '<p>'+'<b>&nbspMatched_pathways</b>: '+dat['match_pathway']+ '</p>'
            information += '<p>'+'<b>&nbspMatch_pwgroup</b>: '+dat['match_pwgroup']+ '</p>'
            information += '<p>'+'<b>&nbspMatch_mts</b>: '+dat['match_mts']+ '</p>'
            information += '<p>'+'<b>&nbspMatch_mtgroups</b>: '+dat['match_mtgroups']+ '</p>'
            information += '<p>'+'<b>&nbspMatch_metastage</b>: '+dat['match_metastage']+ '</p>'
            information += '<br/>'

	    	$(title).appendTo("#title");
	    	$(abstract).appendTo("#abstract");
            $(information).appendTo("#info")

            $('#tobold').keypress(function(e) {
                if (e.keyCode == '13') {
                    e.preventDefault();
                    var $q = $('#abstract');
                    $q.html(
                        $q.html().replace(new RegExp("\\b"+this.value,'gi'), '<span style="background-color: #FFFF00">'+this.value+'</span>')
                    );
                    var $h = $('#title');
                    $h.html(
                        $h.html().replace(new RegExp("\\b"+this.value,'gi'), '<span style="background-color: #FFFF00">'+this.value+'</span>')
                    );
                }
            });
            
            //Only if the user puts in an input, will the text be replaced to highlight
            //if (word) {
    	      //  var $q = $('#abstract');
    		   // $q.html(
    		    //    $q.html().replace(new RegExp("\\b"+word,'gi'), '<span style="background-color: #FFFF00">'+word+'</span>')
    		    //);
    		    //var $h = $('#title');
    		    //$h.html(
    		     //   $h.html().replace(new RegExp("\\b"+word,'gi'), '<span style="background-color: #FFFF00">'+word+'</span>')
    		    //);
            //}

	    });   
    }


    //         console.log(dat['AwardCode'])
    //         console.log(dat['AwardTitle'])
    //         console.log(dat['TechAbstract'])
    //         console.log(dat['Breast Cancer'])
    //         console.log(dat['City'])
    //         console.log(dat['Country'])
    //         console.log(dat['Funding Org'])
    //         console.log(dat['ID'])
    //         console.log(dat['Institution'])


    //         console.log(dat['*Metastasis stage'])
    //         console.log(dat['Metastasis Y/N'])
    //         console.log(dat['Molecular Target'])
    //         console.log(dat['Molecular Target (Group)'])
    //         console.log(dat['PIFirstName'])
    //         console.log(dat['PILastName'])
    //         console.log(dat['Pathway'])
    //         console.log(dat['Pathway (Group)'])

    //     })
    // }
}
