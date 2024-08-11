class Grid {
  constructor(height, width, values) {
    this.height = height;
    this.width = width;
    this.grid = new Array(height);
    for (var i = 0; i < height; i++) {
      this.grid[i] = new Array(width);
      for (var j = 0; j < width; j++) {
        if (
          values != undefined &&
          values[i] != undefined &&
          values[i][j] != undefined
        ) {
          this.grid[i][j] = values[i][j];
        } else {
          this.grid[i][j] = 0;
        }
      }
    }
  }
}

function floodfillFromLocation(grid, i, j, symbol) {
  i = parseInt(i);
  j = parseInt(j);
  symbol = parseInt(symbol);

  target = grid[i][j];

  if (target == symbol) {
    return;
  }

  function flow(i, j, symbol, target) {
    if (i >= 0 && i < grid.length && j >= 0 && j < grid[i].length) {
      if (grid[i][j] == target) {
        grid[i][j] = symbol;
        flow(i - 1, j, symbol, target);
        flow(i + 1, j, symbol, target);
        flow(i, j - 1, symbol, target);
        flow(i, j + 1, symbol, target);
      }
    }
  }
  flow(i, j, symbol, target);
}

function parseSizeTuple(size) {
  size = size.split("x");
  if (size.length != 2) {
    alert('Grid size should have the format "3x3", "5x7", etc.');
    return;
  }
  if (size[0] < 1 || size[1] < 1) {
    alert("Grid size should be at least 1. Cannot have a grid with no cells.");
    return;
  }
  if (size[0] > 30 || size[1] > 30) {
    alert("Grid size should be at most 30 per side. Pick a smaller size.");
    return;
  }
  return size;
}

function convertSerializedGridToGridObject(values) {
  height = values.length;
  width = values[0].length;
  return new Grid(height, width, values);
}

function fitCellsToContainer(
  jqGrid,
  height,
  width,
  containerHeight,
  containerWidth
) {
  candidate_height = Math.floor((containerHeight - height) / height);
  candidate_width = Math.floor((containerWidth - width) / width);
  size = Math.min(candidate_height, candidate_width);
  size = Math.min(MAX_CELL_SIZE, size);
  jqGrid.find(".cell").css("height", size + "px");
  jqGrid.find(".cell").css("width", size + "px");
}

async function getCaptionFromTogether(data) {
  const together_key = localStorage.getItem("together_key");
  const together_model = localStorage.getItem("together_model");

  if (together_key === null || together_model === null) {
    alert("No Together model or key found. Please save first.");
    return;
  }

  const messages = [
    {
      role: "system",
      content:
        "You are an assistant who describes grids in detail, exactly how people see them.",
    },
    {
      role: "user",
      content: JSON.stringify(data),
    },
  ];

  const requestBody = {
    model: together_model,
    messages: messages,
  };

  try {
    // Make the API call using fetch
    const response = await fetch(
      "https://api.together.xyz/v1/chat/completions",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${together_key}`,
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      }
    );

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const jsonResponse = await response.json();

    const caption = jsonResponse.choices[0].message.content;
    return caption;
  } catch (error) {
    console.error("Error:", error);
    alert("An error occurred while fetching the caption.");
    return null;
  }
}

function fillJqGridWithData(jqGrid, dataGrid) {
  jqGrid.empty();
  height = dataGrid.height;
  width = dataGrid.width;
  for (var i = 0; i < height; i++) {
    var row = $(document.createElement("div"));
    row.addClass("row");
    for (var j = 0; j < width; j++) {
      var cell = $(document.createElement("div"));
      cell.addClass("cell");
      cell.attr("x", i);
      cell.attr("y", j);
      setCellSymbol(cell, dataGrid.grid[i][j]);
      row.append(cell);
    }
    jqGrid.append(row);
  }

  //   here, add a button that is called caption, and when clicked, displays a hardcoded caption of the image
  var caption = $(document.createElement("button"));
  caption.addClass("caption");
  caption.text("Caption");

  // add the button to the parent of jqGrid
  // jqGrid.parent().append(caption)
  // add a grid to wrap the jqgrid, and append jq grid into it, and then add that to the parent of jqGrid
  const currentParent = jqGrid.parent();
  const gridWrapper = $(document.createElement("div"));
  gridWrapper.addClass("grid-wrapper");
  gridWrapper.append(jqGrid);
  gridWrapper.append(caption);
  currentParent.append(gridWrapper);

  // set style of grid wrapper
  gridWrapper.css("display", "inline-block");

  const captionText =
    "This is a caption for the image. It is a very nice image.";
  // put it in a p and hide it
  var captionTextElement = $(document.createElement("p"));
  captionTextElement.text(captionText);
  captionTextElement.hide();
  jqGrid.parent().append(captionTextElement);

  // when the button is clicked, show the caption
  caption.click(function () {
    if (captionTextElement.is(":visible")) {
      captionTextElement.hide();
    } else {
      captionTextElement.show();
      // get the caption from together and set it
      const caption = getCaptionFromTogether(dataGrid.grid).then((caption) => {
        captionTextElement.text(caption);
      });
    }
  });
}

function copyJqGridToDataGrid(jqGrid, dataGrid) {
  row_count = jqGrid.find(".row").length;
  if (dataGrid.height != row_count) {
    return;
  }
  col_count = jqGrid.find(".cell").length / row_count;
  if (dataGrid.width != col_count) {
    return;
  }
  jqGrid.find(".row").each(function (i, row) {
    $(row)
      .find(".cell")
      .each(function (j, cell) {
        dataGrid.grid[i][j] = parseInt($(cell).attr("symbol"));
      });
  });
}

function setCellSymbol(cell, symbol) {
  cell.attr("symbol", symbol);
  classesToRemove = "";
  for (i = 0; i < 10; i++) {
    classesToRemove += "symbol_" + i + " ";
  }
  cell.removeClass(classesToRemove);
  cell.addClass("symbol_" + symbol);
  // Show numbers if "Show symbol numbers" is checked
  if ($("#show_symbol_numbers").is(":checked")) {
    cell.text(symbol);
  } else {
    cell.text("");
  }
}

function changeSymbolVisibility() {
  $(".cell").each(function (i, cell) {
    if ($("#show_symbol_numbers").is(":checked")) {
      $(cell).text($(cell).attr("symbol"));
    } else {
      $(cell).text("");
    }
  });
}

function errorMsg(msg) {
  $("#error_display").stop(true, true);
  $("#info_display").stop(true, true);

  $("#error_display").hide();
  $("#info_display").hide();
  $("#error_display").html(msg);
  $("#error_display").show();
  $("#error_display").fadeOut(5000);
}

function infoMsg(msg) {
  $("#error_display").stop(true, true);
  $("#info_display").stop(true, true);

  $("#info_display").hide();
  $("#error_display").hide();
  $("#info_display").html(msg);
  $("#info_display").show();
  $("#info_display").fadeOut(5000);
}
